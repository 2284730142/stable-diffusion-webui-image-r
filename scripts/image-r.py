import gradio as gr
import numpy as np
import cv2
from modules import script_callbacks


def handle_image_contrast(x, c):
    imggray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(x)[0]
    graynew = m + c * (imggray - m)
    img1 = np.zeros(x.shape, np.float32)
    k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
    img1[:, :, 0] = x[:, :, 0] * k
    img1[:, :, 1] = x[:, :, 1] * k
    img1[:, :, 2] = x[:, :, 2] * k
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    return img1.astype(np.uint8)


def handle_image_brightness(x, b):
    # print('b', b)
    [aver_b, aver_g, aver_r] = np.array(cv2.mean(x))[:-1] / 3
    k = np.ones((x.shape))
    k[:, :, 0] *= aver_b
    k[:, :, 1] *= aver_g
    k[:, :, 2] *= aver_r
    x = x + (b - 1) * k
    x[x > 255] = 255
    x[x < 0] = 0
    return x.astype(np.uint8)


def handle_image_gray(x):
    gray_value = 0.07 * x[:, :, 2] + 0.72 * x[:, :, 1] + 0.21 * x[:, :, 0]
    gray_img = gray_value.astype(np.uint8)
    return gray_img


def draw_main():
    with gr.Blocks() as app:
        with gr.Tab("饱和度调整"):
            with gr.Row():
                image_input_contrast = gr.Image()
                image_output_contrast = gr.Image()
            image_slider_contrast = gr.Slider(minimum=0, maximum=2, step=0.05, label='饱和度', interactive=True, value=1)
            image_button_contrast = gr.Button(value='开始处理', variant='primary')
            image_button_contrast.click(fn=handle_image_contrast, inputs=[image_input_contrast, image_slider_contrast], outputs=image_output_contrast)
        with gr.Tab("亮度调整"):
            with gr.Row():
                image_input_brightness = gr.Image()
                image_output_brightness = gr.Image()
            image_slider_brightness = gr.Slider(minimum=0, maximum=2, step=0.05, label='亮度差', interactive=True, value=1)
            image_button_brightness = gr.Button(value='开始处理', variant='primary')
            image_button_brightness.click(fn=handle_image_brightness, inputs=[image_input_brightness, image_slider_brightness], outputs=image_output_brightness)
        with gr.Tab("黑白化【灰度】调整"):
            with gr.Row():
                image_input_gray = gr.Image()
                image_output_gray = gr.Image()
            image_button_gray = gr.Button(value='开始处理', variant='primary')
            image_button_gray.click(fn=handle_image_gray, inputs=image_input_gray, outputs=image_output_gray)


# app.launch()

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as my_images_ride:
        draw_main()
    return (my_images_ride, "图像颜色调整", "my_images_ride"),


script_callbacks.on_ui_tabs(on_ui_tabs)
