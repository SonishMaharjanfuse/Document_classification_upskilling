import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


def upload_image():
    """
    Display a file uploader for image selection.

    Returns:
        Image: PIL Image object.
    """
    uploaded_image = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image


def yolo_predict():
    """
    Perform prediction using YOLO model.
    """
    model = YOLO("./Week_3/model/yolo_model/weights/best.pt")

    # Perform prediction when the user clicks the "Predict" button
    image = upload_image()

    if st.button("Predict"):
        # Convert the PIL image to NumPy array
        cv2_image = np.array(image)
        image_array = (
            cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2BGR)
            if cv2_image.shape[-1] == 4
            else cv2_image
        )
        results = model.predict(image_array)
        name_dic = results[0].names
        key = np.argmax(results[0].probs.data).item()
        st.subheader("Prediction:")
        st.write("Prediction class:", name_dic[key])


def vgg_predict():
    """
    Perform prediction using VGG model.
    """
    model = load_model("./Week_3/model/vgg/vgg16.keras")
    class_names = ['citizenship', 'license', 'others', 'passport']
    image = upload_image()

    # Perform prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        cv2_image = np.array(image)
        image_array = (
            cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2BGR)
            if cv2_image.shape[-1] == 4
            else cv2_image
        )
        # Resize the image
        resized_image = cv2.resize(image_array, (512, 512))

        # Add an extra dimension to match the expected input shape of the model
        input_image = np.expand_dims(resized_image, axis=0)

        # Assuming model is a pre-trained VGG16 model
        predictions = model.predict(input_image)[0]

        # Get the index of the predicted class
        predicted_class_index = np.argmax(predictions)

        # Get the class name based on the index
        predicted_class_name = class_names[predicted_class_index]

        # Display the prediction results
        st.subheader("Prediction:")
        st.write(f"Predicted Class: {predicted_class_name}")
        st.write("Predicted Probabilities:", predictions)


def main():
    options = st.sidebar.radio("Choose the model", ["Home", "YOLO", "VGG"])

    if options == "YOLO":
        yolo_options = st.sidebar.radio(
            "Choose option", ["About model", "Prediction"])
        if yolo_options == "About model":
            st.markdown(
                """
                ### About model
                The YOLOv8 image classification model is designed to detect 1000 pre-defined classes in images in real-time.

                Image classification is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes. Different from YOLO's Segment and Object detection models which are trained on COCO datasets, the image classification models are trained on ImageNet dataset. YOLOv8 comes bundled with image classification models pre-trained on the ImageNet dataset with an image resolution of 224.

                The same model was used for document classification (citizenship, license, others, and passport). For training purpose, we used an augmented dataset that we prepared in week 2. In comparison to other models, YOLO has performed extremely well in our case.
                """
            )
        if yolo_options == "Prediction":
            yolo_predict()

    if options == "VGG":
        vgg_options = st.sidebar.radio(
            "Choose option", ["About model", "Prediction"])
        if vgg_options == "About model":
            st.markdown(
                """
                ### About Model
                The VGG (Visual Geometry Group) model, introduced in 2014, is a notable convolutional neural network architecture designed for image classification. Known for its simplicity and uniform structure, VGGNet employs stacks of 3x3 convolutional filters followed by max-pooling layers. The network is characterized by its depth, with VGG16 comprising 16 layers, including 13 convolutional and 3 fully connected layers, and VGG19 with an additional three convolutional layers. The architecture utilizes rectified linear units (ReLU) as activation functions and concludes with a softmax layer for producing class probabilities.

                VGG achieved remarkable success in the ImageNet Large-Scale Visual Recognition Challenge, showcasing its efficacy in learning hierarchical representations for various object categories. Its design simplicity and effectiveness have rendered VGG a popular choice for image classification tasks, often serving as a pre-trained model for transfer learning. However, its drawback lies in its computational demands due to the large number of parameters. Despite this limitation, VGG's impact on the field of computer vision persists, influencing subsequent architectures and contributing to the broader understanding of deep learning principles in image recognition.
                """
            )
        if vgg_options == "Prediction":
            vgg_predict()

    if options == "Home":
        st.markdown(
            """
            ### Document Classification
            Document classification is a vital task in natural language processing (NLP) and information retrieval, aimed at automatically categorizing textual documents into predefined classes or categories. In the context of document classification for various types of identification documents, such as citizenship, license, passport, and others, the goal is to create a system that can automatically assign the correct label to a given document based on its content.

            In this scenario, the classes—citizenship, license, passport, and others—represent distinct document types with specific information and formats. Document classification models trained for this purpose typically leverage machine learning algorithms, often employing techniques such as text preprocessing, feature extraction, and supervised learning. The models learn patterns and features from labeled examples of each document type to make accurate predictions on unseen documents.

            The application of document classification is significant in automating administrative processes, identity verification, and information retrieval systems. It streamlines the categorization of diverse documents, contributing to more efficient document management workflows, and is particularly relevant in industries where compliance and security play critical roles. Overall, document classification serves as a valuable tool for organizing and handling large volumes of documents with diverse content and purposes.
            """
        )


if __name__ == "__main__":
    main()
