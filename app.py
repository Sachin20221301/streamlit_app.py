# streamlit_app.py
import streamlit as st
from ultralyticsplus import YOLO, render_result

def main():
    st.title("YOLO Object Detection with UltralyticsPlus")
    
    # Load model
    model = YOLO('keremberke/yolov8m-csgo-player-detection')

    # Set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # Set image URL
    image_url = st.text_input("Enter the image URL:")

    # Perform inference
    if image_url:
        results = model.predict(image_url)
        render = render_result(model=model, image=image_url, result=results[0])
        st.image(render.image, use_column_width=True)
        st.write(f"Detected Classes: {', '.join(render.labels)}")

if __name__ == '__main__':
    main()
