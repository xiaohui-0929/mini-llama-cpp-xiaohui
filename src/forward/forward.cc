#include "mini_llama/forward/forward.h"

#include <cstring>
#include <cmath>

#include "mini_llama/tensor_and_Ops/ops.h"
#include "mini_llama/tensor_and_Ops/thread_pool.h"

/**
 * 当前文件内实现 暂不考虑CUDA、量化等场景
 * 部分接口入参暂时未使用 故编译过程中可能会有对应的warning提示
 */

namespace mini_llama {

/**
 * 有效性检查
 */
static void ValidateForwardInputs(const MiniLlamaContext& ctx,
                                  const MiniLlamaModel& model,
                                  int token) {
    const ModelConfig& config = model.config;
    if (!model.loaded) {
        throw std::runtime_error(
            "ForwardToken called with an unloaded model"
        );
    }
    if (token < 0 || token >= config.vocab_size) {
        throw std::out_of_range(
            "ForwardToken token id out of range"
        );
    }
    if (ctx.pos < 0 || ctx.pos >= config.max_seq_len) {
        throw std::out_of_range(
            "ForwardToken position out of range"
        );
    }
    if (model.layers.size() != static_cast<size_t>(config.n_layers)) {
        throw std::runtime_error(
            "ForwardToken layer count does not match model config"
        );
    }
    if (model.token_embedding.data_.size() <
        static_cast<size_t>(config.vocab_size * config.dim)) {
        throw std::runtime_error(
            "ForwardToken token embedding tensor is smaller than config"
        );
    }
}

/**
 * shape检查
 */
static void RequireShape(const Tensor& t, const std::vector<int>& expected,
                         const char* caller) {
    t.AssertShape(expected, caller);
}

/**
 * AddOptionalBias 增加偏置项
 */
static Tensor AddOptionalBias(const Tensor& x, const Tensor& bias,
                              const char* caller) {
    if (bias.data_.empty()) {
        return x;
    }

    if (x.num_dims() != 1 || bias.num_dims() != 1 ||
        x.shape_[0] != bias.shape_[0]) {
        throw std::runtime_error(std::string(caller) +
                                 ": bias shape mismatch x=" + x.ShapeStringShort() +
                                 " bias=" + bias.ShapeStringShort());
    }

    Tensor y = x;
    for (int i = 0; i < x.shape_[0]; ++i) {
        y[i] += bias[i];
    }
    return y;
}

/**
 * Linear 线性变换
 */
static Tensor ForwardLinear(const MiniLlamaModel& model,
                            const std::string& name, const Tensor& x,
                            const Tensor& weight)  {
    return Linear(x, weight);
}

/**
 * Q K V 投影
 */
static void ForwardQkvProjection(const MiniLlamaModel& model,
                                 const std::string& layer_prefix,
                                 const Tensor& h, const LayerWeights& lw,
                                 Tensor& q_flat, Tensor& k_flat,
                                 Tensor& v_flat) {
    const std::string q_name = layer_prefix + "wq";
    const std::string k_name = layer_prefix + "wk";
    const std::string v_name = layer_prefix + "wv";

    // 增加偏置项
    q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq,
                                           "forward_layer q");
    k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, lw.wk), lw.bk,
                           "forward_layer k");
    v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, lw.wv), lw.bv,
                            "forward_layer v");
}

/**
 * RMSNorm 归一化
 */
static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                             const Tensor& weight, float eps) {
    return RmsNorm(x, weight, eps);
}

/**
 * SwiGlu 激活函数
 */
static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                            const Tensor& up) {
    return SwiGlu(gate, up);
}

/**
 * Add 两矩阵相加
 */
static Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
    if (a.shape_ != b.shape_) {
        throw std::runtime_error("forward_add: shape mismatch " +
                                 a.ShapeStringShort() + " vs " +
                                 b.ShapeStringShort());
    }
    Tensor result(a.shape_, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        result.data_[i] = a.data_[i] + b.data_[i];
    }
    return result;
}

/**
 * Softmax 对x进行softmax操作，转换为概率分布
 */
static Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
    return Softmax(x);
}

/**
 * RoPE 对q k进行旋转操作，使内积结果具备相对位置属性
 */
static void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k,
                        int pos, float theta, RopeType rope_type) {
    Rope(q, k, pos, theta, rope_type);
    return; 
}

/**
 * Embedding 映射
 */
static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
    int dim = model.config.dim;
    Tensor result({dim}, 0.0f);

    memcpy(result.data_.data(), &model.token_embedding.data_[token_id * dim],
           dim * sizeof(float)); 

    return result;
}

/**
 * GQA 头转换
 */
static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
    return q_head / (n_heads / n_kv_heads);
}

/**
 * Attention计算
 */
static Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q,
                               const Tensor& k, const Tensor& v, int pos,
                               int layer, KvCache& kv_cache, int n_heads,
                               int n_kv_heads, int head_dim) {
    // 当前的k, v写入kv_cache
    kv_cache.Write(layer, pos, k, v);

    Tensor attn_out({n_heads, head_dim}, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // compute_head处理单个head的计算
    auto compute_head = [&](int h) {
        // GQA分组 确定当前head使用哪个kv_head
        int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

        // 记录QK得分
        std::vector<float> scores_data(pos + 1, 0.0f);
        // 遍历历史token信息 每个历史token都可以计算出对应的QK得分
        // [head_dim] * [head_dim * pos] = [pos]
        for (int t = 0; t <= pos; ++t) {
            const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q.data_[h * head_dim + d] * k_ptr[d];
            }
            scores_data[t] = dot * scale;
        }
        // 1维张量scores 代表当前head下 本pos的token对历史token的QK得分
        Tensor scores({pos + 1}, 0.0f);
        scores.data_ = std::move(scores_data);
        // 将得分转为概率分布
        Tensor probs = ForwardSoftmax(model, scores);

        // scores和历史token的v进行加权求和 得到当前head下 每一个head_dim的信息
        // [pos] * [pos * head_dim] = [head_dim] 对应当前head的特征向量 
        for (int d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
                val += probs.data_[t] * v_ptr[d];
            }
            attn_out.data_[h * head_dim + d] = val;
        }
    };

    // 并行分组计算head
    ParallelFor(n_heads, [&](int begin, int end) {
        for (int h = begin; h < end; ++h) {
            compute_head(h);
        }
    });

    return attn_out;
}

static Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix,
                         const Tensor& h, const LayerWeights& lw) {
    const std::string gate_name = layer_prefix + "w_gate";
    const std::string up_name = layer_prefix + "w_up";
    const std::string down_name = layer_prefix + "w_down";
    // 计算门控张量
    Tensor gate = ForwardLinear(model, gate_name, h, lw.w_gate);
    // 升维
    Tensor up =
            ForwardLinear(model, up_name, h, lw.w_up);  // [hidden_dim]
    // 激活
    Tensor active =
            ForwardSwiGlu(model, gate, up); // [hidden_dim]
    // 降维
    Tensor ff = 
            ForwardLinear(model, down_name, active, lw.w_down); // [dim]

    return ff;
}

/**
 * Transformer 层级计算
 */
static Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           const Tensor& x, int layer, const LayerWeights& lw,
                           const ModelConfig& config) {
    int dim = config.dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_dim = config.head_dim;
    int pos = ctx.pos;
    const std::string layer_perfix = "layers." + std::to_string(layer) + ".";

    // ---- Attention层 ----
    // Pre-Norm 归一化
    // [dim] -> [dim]
    Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, config.rms_norm_eps);

    // Q K V 线性变换
    Tensor q_flat, k_flat, v_flat;
    // q:   [dim] -> [n_heads * head_dim]
    // k/v: [dim] -> [n_kv_heads * head_dim]
    ForwardQkvProjection(model, layer_perfix, h, lw, q_flat, k_flat, v_flat);
    // 按照合适的shape进行变形
    // q:   [n_heads * head_dim]     -> [n_heads, head_dim]
    // k/v: [n_kv_heads * head_dim]  -> [n_kv_heads, head_dim]
    Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
    Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
    Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");

    // RoPE 注入位置信息
    ForwardRope(model, q, k, pos, config.rope_theta, config.rope_type);

    // Attention计算
    // attn_out: [n_heads, head_dim]
    Tensor attn_out =
        AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                         n_heads, n_kv_heads, head_dim);

    // 将attn_out进行reshape 展平为1维张量
    // [n_heads, head_dim] -> [1, n_heads * head_dim]
    Tensor atten_out_flat = attn_out.ReshapeChecked({1, n_heads * head_dim},
                                                    "forward_layer attn_out");
    // 乘以输出权重矩阵Wo 将各个独立的head特征向量进行融合
    // [1, n_heads * head_dim] -> [1, dim]
    Tensor attn_proj =
        ForwardLinear(model, layer_perfix + "wo", atten_out_flat, lw.wo);
    RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj");
    Tensor attn_proj_1d =
        attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");
    // 残差连接 投影结果加上原始输入
    Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

    // ---- FFN层 ----
    // Pre-Norm 归一化
    // [dim] -> [dim]
    Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, config.rms_norm_eps);
    Tensor ff = FfnForward(model, layer_perfix, h2, lw);
    RequireShape(ff, {dim}, "forward_layer ffn");

    return ForwardAdd(model, x_attn, ff);
}

/**
 * 计算词表内各词的得分
 */
static Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                            const ModelConfig& config) {
    // Pre-Norm 归一化
    Tensor normed = ForwardRmsNorm(model, x, model.final_norm, config.rms_norm_eps);
    // 计算分数
    Tensor logits_flat =
        ForwardLinear(model, "lm_head", normed, model.lm_head); // [vocab_size]
    return logits_flat.ReshapeChecked({config.vocab_size}, "compute_logits");
}

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
    // 防御性检查
    ValidateForwardInputs(ctx, model, token);

    const ModelConfig& config = model.config;
    int n_layers = config.n_layers;

    // 1. Embedding 映射 token_id转为对应的一维特征向量
    Tensor x = EmbedToken(model, token);

    // 2. Transformer 层级计算
    for (int layer = 0; layer < n_layers; ++layer) {
        x = ForwardLayer(ctx, model, x, layer, model.layers[layer], config);
    }

    // 3. 计算logits [vocab_size] 代表词表内每个词的分数
    Tensor logits = ComputeLogits(x, model, config);

    return logits;
}

}   // namespace mini_llama