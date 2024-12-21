import time
from pipeline import *
import streamlit as st
import cv2
import os
from skimage import io

directory = "./imgs"
images = os.listdir(directory)

def process_image_for_display(img):
    print(img)
    if img is None:
        return None
    # Convert to float in range [0,1]
    img = img.astype(float) / 255.0
    # Ensure values are clipped to [0,1]
    return np.clip(img, 0, 1)

def main():
    st.title("License Plate Detection")
    st.write("This is a simple web app to detect license plates in images")
    
    # Image selection
    selected_image = st.selectbox("Select an image", images)
    image_path = os.path.join(directory, selected_image)
    # Display selected image
    st.image(image_path, caption="Selected Image")
    # Load and process image
    if st.button("Detect License Plate"):
        # start timer
        start_time = time.time()


        with st.spinner('Processing image...'):
            img = io.imread(image_path)
            img = cv2.resize(img, (704, 576))
            gray_img = rgb2gray(img)
            #return gray_img,filtered_image, equalized_image, binary_image,  roi_img, detected_plate, final_cropped_plate

            gray_img, filtered_image, equalized_image, binary_image, inital_roi_image , roi_img, detected_plate, final_cropped_plate = LicenesePlateDetector(gray_img)
            gray_img = process_image_for_display(gray_img)
            filtered_image = process_image_for_display(filtered_image)
            equalized_image = process_image_for_display(equalized_image)
            roi_img = process_image_for_display(roi_img)
            detected_plate = process_image_for_display(detected_plate)
            final_cropped_plate = process_image_for_display(final_cropped_plate)
            inital_roi_image = process_image_for_display(inital_roi_image)
            end = time.time()

            processing_time = end - start_time
            # Create tabs for different processing stages
            tab1, tab2, tab3 = st.tabs(["Pre-processing", "ROI Detection", "Final Result"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Image")
                with col2:
                    st.image(gray_img, caption="Grayscale Image")
                    
                col3, col4 = st.columns(2)
                with col3:
                    st.image(filtered_image, caption="Filtered Image")
                with col4:
                    st.image(equalized_image, caption="Equalized Image")
                    
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(binary_image, caption="Binary Image")
                with col2:
                    st.image(roi_img, caption="ROI Detection")
                with col3:
                    st.image(inital_roi_image, caption="Detected Plate")
                    
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(detected_plate, caption="Detected Plate")
                with col2:
                    st.image(final_cropped_plate, caption="Final Result")
                
                # End timer
                
                # Add metrics
                st.metric(label="Processing Time", value=f"{processing_time:.2f} seconds")
            
            st.success('License plate detection completed!')
if __name__ == "__main__":
    main()