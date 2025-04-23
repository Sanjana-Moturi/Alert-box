# Alert-box
Notification Manager - Spam/Ham Classifier
This application is designed to manage and classify notifications as either spam or ham. It allows users to add, search, filter, and export notifications, helping to organize and categorize incoming messages effectively.

Problem: With the increasing number of notifications we receive, it's difficult to distinguish between useful messages (ham) and unwanted or suspicious ones (spam). Sorting these manually can be tedious.
Unique Aspect: This idea automates the classification process using machine learning (Naive Bayes with TF-IDF vectorization), which helps categorize notifications effectively without manual intervention.
Improvement: Unlike basic filtering systems, this application uses a combination of machine learning and fuzzy matching to enhance accuracy. Additionally, the ability to attach media and timestamp notifications adds a layer of flexibility that is not typically seen in simple notification managers.

Implementation: The idea is implemented using Python with a combination of libraries such as sklearn, fuzzywuzzy, and ttkbootstrap. A machine learning model is trained to classify notifications as spam or ham based on the message content. A graphical interface is built using Tkinter to allow for easy interaction, and features such as media attachment, search filters, and export/import are included.

Challenges: Some challenges included handling various types of notifications and ensuring that the machine learning model could accurately classify them. Fine-tuning the classification model and ensuring the UI provided a smooth experience were other areas that required attention.

Future Features: It would be beneficial to incorporate a notification prioritization feature, where users can flag important notifications based on certain keywords or patterns. Additionally, integrating push notifications or real-time alerts for spam could be a valuable enhancement.

