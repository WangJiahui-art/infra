# A compilable PyTorch extension framework in order to test cuda kernel
fa_ext/
 - main.cpp          # pybind11 绑定
 - test.py           # 验证脚本
 - build.py          # torch.utils.cpp_extension.load 编译脚本