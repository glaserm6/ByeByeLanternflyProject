## Milestone 3: Model Development and Evaluation

### 1. Introduction

The goal of Bye-Bye Lanternfly is to develop a computer vision application that identifies and counts adult spotted lanternflies (*Lycorma delicatula*) in user-submitted photographs. The project is intended to help users assess the presence of adult lanternflies on trees and other surfaces.

For Milestone 3, the primary focus was developing and evaluating the AI component of the application. The project uses a YOLOv8 object detection model trained to recognize adult spotted lanternflies. The trained model is integrated into a Flask web application, allowing users to upload an image and receive an annotated image and estimated lanternfly count.

### 2. Dataset / Problem Environment

The model was trained using an adult spotted lanternfly image dataset from Roboflow Universe (Vasudevs v7). The dataset was filtered and prepared to focus on adult lanternflies rather than other life stages.

The final dataset used for training contained:

| Dataset Split | Images | Bounding Boxes |
|---|---:|---:|
| Training | 1,827 | 2,978 |
| Validation | 438 | 788 |
| Testing | 276 | 765 |
| **Total** | **2,541** | **4,531** |

Each lanternfly was annotated with a bounding box and assigned to a single detection class: `lanternfly`.

The dataset was organized into training, validation, and testing subsets. Additional preparation included checking annotations, filtering questionable images, and correcting dataset formatting to ensure compatibility with YOLOv8.

### 3. Methodology

The project uses YOLOv8n, a pretrained object detection model from Ultralytics. Rather than developing an object detection algorithm from scratch, the project uses transfer learning to adapt an existing model to recognize adult spotted lanternflies.

The model was trained using Google Colab with a GPU runtime. The primary training configuration included:

- **Model:** YOLOv8n
- **Training epochs:** 100
- **Image size:** 640 × 640
- **Batch size:** 16
- **Optimizer:** Ultralytics default configuration
- **Early stopping patience:** 20 epochs
- **Hardware:** NVIDIA Tesla T4 GPU
- **Random seed:** 42

After training, the best-performing model checkpoint (`best.pt`) was saved and integrated into the Flask application.

The application uses the trained model to identify lanternflies in uploaded photographs. It applies a confidence threshold of 0.20 and an intersection-over-union (IoU) threshold of 0.45 during detection. The application then displays the annotated image and the total number of detected lanternflies.

### 4. Experimental Setup

The model was evaluated using the held-out testing dataset. Evaluation focused on how accurately the model could identify and count adult lanternflies in individual images.

The primary evaluation metrics included:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between the actual and predicted lanternfly counts.
- **Root Mean Squared Error (RMSE):** Measures the magnitude of counting errors while placing greater emphasis on larger errors.
- **Exact Count Accuracy:** Measures the percentage of images for which the predicted count exactly matches the actual count.
- **Within-One Accuracy:** Measures the percentage of images for which the predicted count is within one lanternfly of the actual count.

The evaluation also examined overcounting and undercounting to identify situations in which the model struggled.

### 5. Preliminary Results

The final 100-epoch training run produced a trained YOLOv8n model (`best.pt`). Evaluation on the held-out testing dataset produced the following results:

| Evaluation Metric | Result |
|---|---:|
| Testing images | 276 |
| Actual lanternflies | 765 |
| Predicted lanternflies | 696 |
| Mean Absolute Error (MAE) | 0.4674 |
| Root Mean Squared Error (RMSE) | 3.1027 |
| Exact Count Accuracy | 89.13% |
| Within-One Accuracy | 95.29% |
| Images with overcounts | 17 |
| Images with undercounts | 13 |

Overall, the model demonstrated promising counting performance on the testing dataset. It predicted the exact number of lanternflies in approximately 89% of test images and produced a count within one lanternfly of the actual count in approximately 95% of images.

However, the RMSE was considerably higher than the MAE, indicating that a small number of images contained substantially larger counting errors. The largest observed undercount was 42 lanternflies.

### 6. Initial Discussion

The experimental results suggest that YOLOv8n can effectively identify and count adult spotted lanternflies in many of the images included in the testing dataset. The relatively low MAE and high within-one accuracy indicate that the model generally performs well when estimating lanternfly counts.

However, the results also demonstrate that overall performance can be misleading when a small number of images produce substantial errors. In particular, the largest undercount of 42 indicates that the model may struggle with certain image conditions.

Potential contributing factors include:

- Lanternflies appearing close together or overlapping.
- Unusual lighting conditions or shadows.
- Small or partially obscured lanternflies.
- Differences in image quality, camera distance, and orientation.
- Images containing unusually large numbers of lanternflies.

One important lesson from this experiment is that object detection performance should be evaluated using more than a single metric. Although exact count accuracy and MAE were encouraging, RMSE helped reveal significant errors that would otherwise be less apparent.

The current model is a promising starting point, but additional testing with photographs taken outside the original dataset is necessary to determine how well it generalizes to real-world conditions.

### 7. Remaining Work

Before the final project presentation, the following tasks will be prioritized:

1. **Test with additional real-world photographs:** Evaluate the trained model using photographs taken with a personal smartphone under different lighting conditions, distances, and levels of lanternfly density.
2. **Improve the dataset:** Add additional photographs and manually annotate new examples to improve the model's ability to recognize lanternflies in challenging conditions.
3. **Investigate significant counting errors:** Examine images with large undercounts and determine whether additional training data, annotation improvements, or parameter adjustments could improve performance.
4. **Refine the web application:** Continue improving the Flask interface, image upload process, annotated image display, and counting results.
5. **Finalize evaluation documentation:** Organize the training results, evaluation metrics, and example predictions for the final report and presentation.

The overall objective is to produce a functional, easy-to-use application that demonstrates the practical use of computer vision for identifying and counting adult spotted lanternflies.

### 8. References

- Ultralytics. *YOLOv8 Documentation*. https://docs.ultralytics.com/
- Roboflow Universe. *Vasudevs Spotted Lanternfly Dataset, Version 7*. Adult spotted lanternfly image dataset, licensed under CC BY 4.0.
- Google Colab. *Hosted Jupyter Notebook Service*. https://colab.research.google.com/