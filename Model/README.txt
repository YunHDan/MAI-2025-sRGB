First you need to create a new virtual environment:

conda create convert python=3.10
conda activate convert
pip install -r requirements.txt

Notes:
- net_g_latest.pth is the pytorch model we trained.
- onnx_convert.py converts pytorch weights to onnx and it supports dynamic input sizes.
- convert_ipynb is the jupyter code in colab where you can open the colab link. The code in this file converts onnx files to tflite files that support dynamic size input, and then you can get model_none.tflite.
- We first convert the pytorch model to an onnx model and then to model_none.tflite

Convert pytorch models to tflite models with fixed input size using ai-edge-torch. The following command needs to be executed:

python Convert.py

Then you can get the model.tflite