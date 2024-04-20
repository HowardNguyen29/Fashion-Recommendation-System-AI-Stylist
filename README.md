# Fashion-Recommendation-System-AI-Stylist
Building A Fashion Recommendation System &amp; AI Stylist end-to-end

Problem Statement:
When shopping for fashion items online, consumers often experience an overwhelming number of choices, leading to "choice overload." This extensive array of options can result in decision fatigue, making it challenging for shoppers to select items confidently. Furthermore, consumers frequently struggle to pair clothing and accessories together effectively, which complicates the process of creating cohesive and satisfying outfits.

Solutions:
Our machine learning product addresses these issues by providing personalized recommendations tailored to individual user preferences. Utilizing advanced algorithms such as k-nearest neighbors, ResNet50, and Gemini, our product simplifies the selection process by reducing decision fatigue. 

ML Canvas:

<img width="417" alt="image" src="https://github.com/HowardNguyen29/Fashion-Recommendation-System-AI-Stylist/assets/144277909/4c51c532-c817-4fd5-8cbe-1d6e9a844bc1">

Model Evaluation:
The dataset does not include labels or categories, evaluating the recommendation system becomes challenging because we don't have ground truth information to compare the recommendations. Here are some approaches we considered:
+ Embedding Similarity: We compute the cosine similarity between embeddings of recommended items and the query item. We were using other dataset which has more than 5000 images about clothing.
+ User Feedback Integration: Integrate user feedback into the recommendation system, such as thumbs-up or thumbs-down buttons, star ratings, or comments. Use this feedback from users to improve the recommendation algorithms.
+ 
![image](https://github.com/HowardNguyen29/Fashion-Recommendation-System-AI-Stylist/assets/144277909/6984ff31-46cc-423a-a2b4-ae58009f415d)

Model Deployment:
![full_stack](https://github.com/HowardNguyen29/Fashion-Recommendation-System-AI-Stylist/assets/144277909/565bc524-57a9-4fb3-bedf-e9422e6f94da)


Challenges:
We were facing a problem with evaluation methods. However, by launching the products, we can enhance our recommendation system by using feedback from our users. AI Stylist Decision will support users to get general decisions from the AI, however, the final decision would be from the users not the AI Decision. Deployment phase also faced a problem as our dataset has tons of quality images, so it takes a lot of time to deploy to docker and to cloud.

Demo:

![Demo-Fashion-Recommendation-System-AI-Stylist](https://github.com/HowardNguyen29/Fashion-Recommendation-System-AI-Stylist/assets/144277909/7e4e5706-ad0a-4ce1-b28e-fc6a2f16a4d3)

Phone version:

![Demo-on-phone](https://github.com/HowardNguyen29/Fashion-Recommendation-System-AI-Stylist/assets/144277909/f3684b5c-679b-44ac-87af-581726a88a6f)


Thank you for your time !!!
