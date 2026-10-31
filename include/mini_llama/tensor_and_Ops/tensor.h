#ifndef INCLUDE_MINI_LLAMA_TENSOR_H_
#define INCLUDE_MINI_LLAMA_TENSOR_H_

#include <vector>
#include <string>
#include <stdexcept>

namespace mini_llama {

struct Tensor {
    std::vector<float> data_;
    std::vector<int> shape_;

    Tensor() = default;
    explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

    // 数据量
    size_t size() const { return data_.size(); }
    size_t num_elements() const { return data_.size(); }

    // 维度
    size_t num_dims() const { return shape_.size(); }

    // 获取指定位置的元素索引
    size_t FlatIndex(const std::vector<int>& indices) const;

    // 元素访问
    float& At(const std::vector<int>& indices);
    float At(const std::vector<int>& indices) const;

    // 便捷访问
    float& At1(int i);
    float At1(int i) const;
    float& At2(int i, int j);
    float At2(int i, int j) const;
    float& At3(int i, int j, int k);
    float At3(int i, int j, int k) const;
    float& At4(int i, int j, int k, int l);
    float At4(int i, int j, int k, int l) const;

    // 获取指定行
    float* RowPtr(int row);
    const float* RowPtr(int row) const;

    // 检查当前tensor的维度信息和传入维度是否一致
    void AssertShape(const std::vector<int>& expected, const char* caller) const;

    // 将当前tensor按照新shape进行变化（需保持data数量一致）
    Tensor ReshapeChecked(const std::vector<int>& new_shape, const char* caller) const;

    // 打印shape
    std::string ShapeStringShort() const;
    std::string ShapeString() const { return ShapeStringShort(); }
    void Print(const std::string& name = "", bool print_data = false) const;

    // []运算符重载 获取指定索引的data
    float& operator[](size_t i) { 
        if (i < data_.size()) {
            return data_[i]; 
        } else {
            throw std::out_of_range("Index out of range in operator[]");
        }
        
    }
    float operator[](size_t i) const { 
        if (i < data_.size()) {
            return data_[i]; 
        } else {
            throw std::out_of_range("Index out of range in operator[]");
        }
    }
};

// Create a 1D tensor
Tensor MakeTensor1D(int d0, float fill = 0.0f);

// Create a 2D tensor
Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);

// Create a 3D tensor
Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);

// Create a 4D tensor
Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TENSOR_H_