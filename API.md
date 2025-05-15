## AI-Related APIs
### Classification by Core AI Function
<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>API</th>
      <th>Description</th>
      <th>Auth</th>
      <th>HTTPS</th>
      <th>CORS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Classification & Regression</td>
      <td><a href="https://bigml.com" target="_blank" rel="noopener">BigML</a></td>
      <td>Supports training and prediction of models such as Decision Trees, Ensembles, Logistic Regression, Linear Regression, Deepnets, etc. Provides batch and real-time prediction capabilities, and can be exported to local low-latency calls.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://aws.amazon.com/cn/sagemaker/?nc=sn&loc=0&refid=8987dd52-6f33-407a-b89b-a7ba025c913c" target="_blank" rel="noopener">SageMaker</a></td>
      <td>Built-in Linear Learner algorithms efficiently train binary or multivariate classification and regression models and automatically scale up and down to hosted endpoints.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">AWS</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <td><a href="https://cloud.google.com/automl" target="_blank" rel="noopener">AutoML</a></td>
      <td>Create classification and regression models for tabular data through low-code interface or REST API, automatic feature engineering and hyperparameter tuning, and one-click deployment of online prediction.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">OAuth/apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2">Clustering & Dimensionality Reduction</td>
      <td><a href="https://bigml.com" target="_blank" rel="noopener">BigML</a></td>
      <td>Supports K-means, G-means clustering, and PCA (Principal Component Analysis) dimensionality reduction, all of which can be invoked through the BigML REST API, and can be exported to run locally.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://azure.microsoft.com/en-us/products/machine-learning?utm_source=chatgpt.com" target="_blank" rel="noopener">AutoML</a></td>
      <td>Multiple clustering algorithms (e.g., KMeans, DBSCAN) and dimensionality reduction (PCA, t-SNE) pipelines are supported through SDK/REST interfaces that can be integrated into the ML Pipeline</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Detection & Recognition</td>
      <td><a href="https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html?utm_source=chatgpt.com" target="_blank" rel="noopener">Rekognition</a></td>
      <td>Provides visual analysis capabilities such as face detection, face comparison, object and scene recognition, text detection, adult content filtering, etc., applicable to image and video</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">AWS</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://www.clarifai.com" target="_blank" rel="noopener">Clarifai</a></td>
      <td>Supports generic image classification, object detection, instance segmentation, face attribute analysis, custom model training and management</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://cloudmersive.com" target="_blank" rel="noopener">Cloudmersive</a></td>
      <td>Includes automatic image description (Captioning), OCR, NSFW detection, face recognition, etc., using a simple apiKey authentication</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Generation & Creation</td>
      <td><a href="https://platform.openai.com/docs/api-reference/images" target="_blank" rel="noopener">OpenAI</a></td>
      <td>A large-scale language model based on the GPT family, supporting tasks such as dialog generation, code completion, article continuation, summarization, etc., quickly invoked via API Key</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://cloud.google.com/generative-ai-studio?hl=zh_cn" target="_blank" rel="noopener">Vertex AI</a></td>
      <td>Provide one-stop UI and API access for text and code generation (Codey), image generation (Imagen), multimodal agent construction, etc.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <td><a href="https://docs.cohere.com/v2/docs/chat-api" target="_blank" rel="noopener">Cohere</a></td>
      <td>Text generation service for developers with fast inference and customizable generation models (requires introduction of Cohere API Key).</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Translation & Multilingualism</td>
      <td><a href="https://cloud.google.com/translate?hl=zh_cn" target="_blank" rel="noopener">Translation AI</a></td>
      <td>Provides automated translations in hundreds of languages, supports batch or real-time requests, high availability with global nodes.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey/OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://www.deepl.com/en/pro-api?utm_source=chatgpt.com" target="_blank" rel="noopener">DeepL</a></td>
      <td>Known for its high quality translations, it supports REST interface calls for text and whole document translations (PDF, Word, PowerPoint).</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <td><a href="https://azure.microsoft.com/en-us/solutions/migration/migrate-modernize-innovate?ef_id=_k_CjwKCAjwuIbBBhBvEiwAsNypvWATxWYlzqlwQYwcHYcnX_oq7RLE5eSgl7w3bDpPnufr9pYhtGtkHhoCpkcQAvD_BwE_k_&OCID=AIDcmme9zx2qiz_SEM__k_CjwKCAjwuIbBBhBvEiwAsNypvWATxWYlzqlwQYwcHYcnX_oq7RLE5eSgl7w3bDpPnufr9pYhtGtkHhoCpkcQAvD_BwE_k_&gad_source=1&gad_campaignid=21040127943&gbraid=0AAAAADcJh_sCGUi16fMkmBULDOT43DhJn&gclid=CjwKCAjwuIbBBhBvEiwAsNypvWATxWYlzqlwQYwcHYcnX_oq7RLE5eSgl7w3bDpPnufr9pYhtGtkHhoCpkcQAvD_BwE" target="_blank" rel="noopener">Azure</a></td>
      <td>Azure provides translation services that support text, voice, and document translation and are compatible with multiple platforms.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey/OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Dialogue & QA</td>
      <td><a href="https://platform.openai.com/docs/guides/conversation-state?api-mode=responses" target="_blank" rel="noopener">OpenAI</a></td>
      <td>Designed for conversational scenarios with multiple rounds of contextualized Q&A via messages arrays, suitable for Chatbot, Virtual Assistants.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://aws.amazon.com/cn/pm/lex/?trk=a56d9498-691a-4818-aa2b-68136069b9ff&sc_channel=ps&ef_id=CjwKCAjwuIbBBhBvEiwAsNypvT4ksAYsEDDZiOZhyeLejpjj0MKktozTWTeb5tvAMWaVJSeswl9d9RoC2GYQAvD_BwE:G:s&s_kwcid=AL!4422!3!650384424957!e!!g!!amazon%20lex!19606019227!145110228843&gad_campaignid=19606019227&gbraid=0AAAAADjHtp-VYJtp3XiVqBKhPJSq8WYKG&gclid=CjwKCAjwuIbBBhBvEiwAsNypvT4ksAYsEDDZiOZhyeLejpjj0MKktozTWTeb5tvAMWaVJSeswl9d9RoC2GYQAvD_BwE" target="_blank" rel="noopener">Amazon Lex</a></td>
      <td>AWS-hosted conversational interface building service with support for speech and text input, automated NLU parsing and intent management.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">AWS</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <td><a href="https://cloud.google.com/conversational-ai?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-b-dr-1710134&utm_content=text-ad-none-any-DEV_c-CRE_736900402018-ADGP_Hybrid+%7C+BKWS+-+MIX+%7C+Txt-AI+and+Machine+Learning-Conversational+AI-KWID_43700081504370537-kwd-1666564570044&utm_term=KW_google+cloud+conversational+ai-ST_google+cloud+conversational+ai&gclsrc=aw.ds&gad_source=1&gad_campaignid=22024911514&gclid=CjwKCAjwuIbBBhBvEiwAsNypvcGJnu8fmN4UqAMrn-uUWRB9Sq8Vow6t_ZmDEoSlKUFXQ3_-ec3e1hoCyO0QAvD_BwE&hl=zh_cn" target="_blank" rel="noopener">Vertex AI</a></td>
      <td>Natural Language Understanding and Conversation Management Platform powered by Google Cloud to build multi-round conversational intelligent bots via REST APIs.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Recommendations & Personalization</td>
      <td><a href="https://cloud.google.com/use-cases/recommendations?hl=zh_cn" target="_blank" rel="noopener">Vertex AI Search</a></td>
      <td>Zero-based one-click access to the recommendation system services, support for e-commerce, media and other scenarios of real-time personalized recommendations, providing REST and gRPC APIs</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey/OAuth</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://aws.amazon.com/cn/personalize/" target="_blank" rel="noopener">Amazon Personalize</a></td>
      <td>Hosted recommendation service from AWS with user behavior import, model training, and real-time recommendation endpoints, compatible with common data formats.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">AWS</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <td><a href="https://www.algolia.com/doc/guides/algolia-recommend/overview/" target="_blank" rel="noopener">Algolia Recommend</a></td>
      <td>Recommendations module on Algolia, a search-as-a-service platform that combines user search and click behavior to provide personalized recommendations.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Search & Retrieve</td>
      <td><a href="https://www.pinecone.io/?utm_source=chatgpt.com" target="_blank" rel="noopener">Pinecone</a></td>
      <td>Professional vector retrieval database with high-performance APIs for similarity search, filtering, real-time indexing, etc., suitable for RAG, semantic retrieval.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://aws.amazon.com/cn/personalize/" target="_blank" rel="noopener">Elastic</a></td>
      <td>Hosted recommendation service from AWS with user behavior import, model training, and real-time recommendation endpoints, compatible with common data formats.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <td><a href="https://www.algolia.com/doc/guides/algolia-recommend/overview/" target="_blank" rel="noopener">Weaviate</a></td>
      <td>Open source vector database, providing GraphQL and REST interfaces, built-in text embedding model and contextual search functions.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">RL & Decision Optimization</td>
      <td><a href="https://docs.aws.amazon.com/sagemaker/latest/dg/reinforcement-learning.html" target="_blank" rel="noopener">SageMaker</a></td>
      <td>SageMaker Reinforcement Learning hosts multiple RL frameworks (e.g., Ray RLlib, Stable Baselines) to support distributed training and hyperparametric search.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">AWS</code></td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <!-- 不需要在这里再写跨行的单元格 -->
      <td><a href="https://platform.openai.com/docs/guides/rft-use-cases" target="_blank" rel="noopener">OpenAI RFT</a></td>
      <td>Enhanced fine-tuning by applying custom feedback signals on the base model to optimize policy generation for specific tasks.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">apiKey</code></td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <td><a href="https://cloud.google.com/vertex-ai/docs/training/using-hyperparameter-tuning?hl=zh-cn" target="_blank" rel="noopener">Vertex AI</a></td>
      <td>Accelerate hyperparametric tuning of RL algorithms or other models by parallelizing jobs with intelligent search algorithms.</td>
      <td><code style="background:#f0f0f0; padding:2px 4px; border-radius:4px;">OAuth</code></td>
      <td>YES</td>
      <td>Unknown</td>
    </tr>
  </tbody>
</table>
