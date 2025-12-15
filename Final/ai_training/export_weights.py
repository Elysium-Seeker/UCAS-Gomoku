import torch
import os
import sys

# 导入模型定义
try:
    from train import GomokuNet
except ImportError:
    # 如果直接运行此脚本，可能需要添加路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train import GomokuNet

def export_weights():
    model_path = "ai_training/gomoku_model.pth"
    output_path = "src/model_weights.h"
    
    if not os.path.exists(model_path):
        print(f"未找到模型文件 {model_path}，将生成随机权重用于测试。")
        model = GomokuNet()
    else:
        print(f"正在加载模型: {model_path}")
        # 强制使用 CPU 加载，避免导出时出现 CUDA 错误
        device = torch.device('cpu')
        model = GomokuNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    print("正在导出权重到 C 头文件 (这可能需要几秒钟)...")
    
    # Get weights as numpy arrays (CPU)
    c1_w = model.conv1.weight.detach().cpu().numpy()
    c1_b = model.conv1.bias.detach().cpu().numpy()
    
    c2_w = model.conv2.weight.detach().cpu().numpy()
    c2_b = model.conv2.bias.detach().cpu().numpy()
    
    fc1_w = model.fc1.weight.detach().cpu().numpy().T
    fc1_b = model.fc1.bias.detach().cpu().numpy()
    
    fc2_w = model.fc2.weight.detach().cpu().numpy().T
    fc2_b = model.fc2.bias.detach().cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 直接写入文件，避免构建巨大的字符串导致内存溢出或极慢
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        
        f.write("// CNN 结构参数 (Enhanced)\n")
        f.write("#define BOARD_SIZE 15\n\n")
        
        f.write("// Layer 1: Conv 5x5, 1->32\n")
        f.write("#define C1_IN 1\n")
        f.write("#define C1_OUT 32\n")
        f.write("#define C1_K 5\n\n")
        
        f.write("// Layer 2: Conv 3x3, 32->64\n")
        f.write("#define C2_IN 32\n")
        f.write("#define C2_OUT 64\n")
        f.write("#define C2_K 3\n\n")
        
        f.write("// FC Layers\n")
        f.write("#define FLATTEN_SIZE (C2_OUT * BOARD_SIZE * BOARD_SIZE) // 64*15*15 = 14400\n")
        f.write("#define FC1_SIZE 256\n\n")

        # Export Conv1
        f.write("// --- Conv1 Weights [32][1][5][5] ---\n")
        f.write("static const float c1_weights[C1_OUT][C1_IN][C1_K][C1_K] = {\n")
        for o in range(32):
            f.write("    {\n")
            for i in range(1):
                f.write("        {\n")
                for r in range(5):
                    f.write("            {")
                    f.write(", ".join([f"{c1_w[o][i][r][c]:.6f}f" for c in range(5)]))
                    f.write("},\n")
                f.write("        },\n")
            f.write("    },\n")
        f.write("};\n\n")

        f.write("static const float c1_bias[C1_OUT] = {\n    ")
        f.write(", ".join([f"{val:.6f}f" for val in c1_b]))
        f.write("\n};\n\n")

        # Export Conv2
        f.write("// --- Conv2 Weights [64][32][3][3] ---\n")
        f.write("static const float c2_weights[C2_OUT][C2_IN][C2_K][C2_K] = {\n")
        for o in range(64):
            f.write("    {\n")
            for i in range(32):
                f.write("        {\n")
                for r in range(3):
                    f.write("            {")
                    f.write(", ".join([f"{c2_w[o][i][r][c]:.6f}f" for c in range(3)]))
                    f.write("},\n")
                f.write("        },\n")
            f.write("    },\n")
        f.write("};\n\n")

        f.write("static const float c2_bias[C2_OUT] = {\n    ")
        f.write(", ".join([f"{val:.6f}f" for val in c2_b]))
        f.write("\n};\n\n")

        # Export FC1
        print("正在写入 FC1 权重 (这可能需要一点时间)...")
        f.write("// --- FC1 Weights [14400][256] ---\n")
        f.write("static const float fc1_weights[FLATTEN_SIZE][FC1_SIZE] = {\n")
        for i in range(14400):
            f.write("    {")
            f.write(", ".join([f"{fc1_w[i][j]:.6f}f" for j in range(256)]))
            f.write("},\n")
            if (i + 1) % 2000 == 0:
                print(f"  已处理 {i+1}/14400 行...")
        f.write("};\n\n")
        
        f.write("static const float fc1_bias[FC1_SIZE] = {\n    ")
        f.write(", ".join([f"{val:.6f}f" for val in fc1_b]))
        f.write("\n};\n\n")

        # Export FC2
        f.write("// --- FC2 Weights [256][1] ---\n")
        f.write("static const float fc2_weights[FC1_SIZE][1] = {\n")
        for i in range(256):
            f.write(f"    {{ {fc2_w[i][0]:.6f}f }},\n")
        f.write("};\n\n")

        f.write(f"static const float fc2_bias[1] = {{ {fc2_b[0]:.6f}f }};\n")

        f.write("\n#endif // MODEL_WEIGHTS_H\n")
    
    print(f"权重已成功导出到: {output_path}")

if __name__ == "__main__":
    export_weights()
