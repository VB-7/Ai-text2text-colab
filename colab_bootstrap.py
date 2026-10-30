# 1 CELL Colab
%pip install -U git+https://github.com/VB-7/Ai-text2text-colab.git
from airi import init_models, launch_ui

# pilih model dasar: "gemma12b" / "gemma_e4b" / "gemma_e2b" / "qwen9b" / "qwen4b" / "qwen2b"
init_models("gemma12b")
launch_ui()
