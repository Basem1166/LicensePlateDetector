import time
from pipeline import *
import streamlit as st
import cv2
import os
from skimage import io

directory = "./dataset/images"
images = os.listdir(directory)

def process_image_for_display(img):
    # print(img)
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
    # Button to go to the next image
   
   
    # Load and process image
    if st.button("Detect License Plate"):
        # start timer
        start_time = time.time()


        with st.spinner('Processing image...'):
            img = io.imread(image_path)
            img = cv2.resize(img, (704, 576))
            #return gray_img,filtered_image, equalized_image, binary_image,  roi_img, detected_plate, final_cropped_plate
            try:
                gray_img, filtered_image, equalized_image, binary_image, inital_roi_image ,filtered_roi_img, roi_img, detected_plate, final_cropped_plate, extracted_plate, characters,plate_text, result = LicenesePlateDetector(img)
            except:
                st.error("No License Plate Detected")
                return
            gray_img = process_image_for_display(gray_img)
            filtered_image = process_image_for_display(filtered_image)
            equalized_image = process_image_for_display(equalized_image)
            roi_img = process_image_for_display(roi_img)
            detected_plate = process_image_for_display(detected_plate)
            final_cropped_plate = process_image_for_display(final_cropped_plate)
            inital_roi_image = process_image_for_display(inital_roi_image)
            filtered_roi_img = process_image_for_display(filtered_roi_img)
            # extracted_plate = process_image_for_display(extracted_plate)
                
            end = time.time()

            processing_time = end - start_time
            # Create tabs for different processing stages
            tab1, tab2, tab3, tab4 = st.tabs(["Pre-processing", "ROI Detection", "License Plate Detection", "Character Recognition"])
            
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
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.image(binary_image, caption="Binary Image")
                with col2:
                    st.image(inital_roi_image, caption="Initial ROI Regions")
                with col3:
                    st.image(filtered_roi_img, caption="Final ROI Regions")
                with col4:
                    st.image(roi_img, caption="Most Promising ROI")
                    
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(detected_plate, caption="Detected Plate")
                with col2:
                    st.image(final_cropped_plate, caption="Final Cropped Plate")
            
            with tab4:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Detected Plate using DNN: {result}")
                with col2:
                    st.write(f"Segmented characters : {plate_text}")
                    for i, char in enumerate(characters):
                        st.image(char, caption=f"Character {i+1}")

            
            st.metric(label="Processing Time", value=f"{processing_time:.2f} seconds")
            
            st.success('License plate detection completed!')
if __name__ == "__main__":
    main()