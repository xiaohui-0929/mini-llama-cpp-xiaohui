
#include <iostream>
#include <cstring>
#include <filesystem>

#include "mini_llama/CLI/generate_mode.h"
#include "mini_llama/CLI/run_mode.h"
#include "mini_llama/model/loader.h"

namespace mini_llama {
/**
 * 打印Usage内容
 */
static void PrintGlobalUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <command> [options]\n"
              << "Commands:\n"
              << "  generate      One-shot text generation (default)\n"
              << "  run           Interactive chat mode\n"
              << "  inspect       Inspect model metadata\n"
            //   << "  inspect-gguf  Inspect GGUF file\n"
            //   << "  bench         Benchmark inference performance\n"
            //   << "  --cuda-info   Print CUDA device information for the selected "
            //      "CUDA build\n"
            //   << "\n"
              << "Run \"" << prog << " generate --help\", \"" << prog
              << " run --help\", or \"" << prog
              << " bench --help\" for details.\n";
}

static void PrintInspectUsage(const char* prog) {
  std::cout << "Usage: " << prog << " inspect <model-path|dir> [options]\n"
            << "Options:\n"
            // << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
            //    "requires -DMINI_LLAMA_CUDA=ON)\n"
            // << "  --device <n>         CUDA device id for --backend cuda "
            //    "(default: 0)\n"
            << "  -h, --help           Show this help\n";
}

static MiniLlamaModel LoadModelForInspect(const std::string& path) {
  std::string model_path = path;
  if (std::filesystem::is_directory(path)) {
    std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
    std::filesystem::path json_path = std::filesystem::path(path) / "model.json";
    if (std::filesystem::exists(bin_path) &&
        std::filesystem::exists(json_path)) {
      return LoadModel(json_path.string(), bin_path.string());
    }
  }

  std::filesystem::path model_file(model_path);
  std::string config_path = (model_file.parent_path() / "model.json").string();
  return LoadModel(config_path, model_path);
}

static int RunInspect(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintInspectUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model path.\n";
    PrintInspectUsage(argv[0]);
    return 1;
  }

  std::string model_path = argv[2];

  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintInspectUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintInspectUsage(argv[0]);
      return 1;
    }
  }

  if (std::filesystem::is_directory(model_path)) {
    std::filesystem::path config_path =
        std::filesystem::path(model_path) / "model.json";
    if (std::filesystem::exists(config_path) &&
        !mini_llama::InspectModel(config_path.string())) {
      return 1;
    }
  }

  MiniLlamaModel model = LoadModelForInspect(model_path);
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  std::cout << "backend: " << "cpu" << "\n";

  return 0;
}

}  // namespace mini_llama

int main(int argc, char** argv) {
    if (argc < 2) {
        mini_llama::PrintGlobalUsage(argv[0]);
        return 1;
    }

    // 获取指令
    std::string command = argv[1];
    if (command == "generate") {
      return mini_llama::RunGenerate(argc, argv);
    } else if (command == "run") {
      return mini_llama::RunChat(argc, argv);
    } else if (command == "inspect") {
      return mini_llama::RunInspect(argc, argv);
    } else if (command == "-h" || command == "--help") {
      mini_llama::PrintGlobalUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown command: " << command << "\n";
      mini_llama::PrintGlobalUsage(argv[0]);
      return 1;
    }

    return 0;
}