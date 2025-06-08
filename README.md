<h1>Deceit - Detector</h1>

<p>This project is a simple web application built using <strong>Streamlit</strong> and <strong>Scikit-learn</strong> that performs credit card fraud detection using a <strong>Logistic Regression</strong> model.</p>

<h2>📂 Project Structure</h2>

<ul>
  <li><code>app.py</code> - Main Streamlit application.</li>
  <li><code>creditcard.csv</code> - Dataset used for training and testing the model (Kaggle dataset).</li>
  <li><code>requirements.txt</code> - Python dependencies.</li>
</ul>

<h2>⚙️ Installation</h2>

<ol>
  <li>Clone this repository:</li>
  <pre><code>git clone https://github.com/yourusername/deceit-detector.git</code></pre>

  <li>Navigate to the project directory:</li>
  <pre><code>cd deceit-detector</code></pre>

  <li>Create a virtual environment (optional but recommended):</li>
  <pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate</code></pre>

  <li>Install required packages:</li>
  <pre><code>pip install -r requirements.txt</code></pre>

  <li>Make sure <code>creditcard.csv</code> is in the same directory as <code>app.py</code>.</li>

  <li>Run the Streamlit app:</li>
  <pre><code>streamlit run app.py</code></pre>
</ol>

<h2>📊 Model</h2>

<p>The application trains a logistic regression model on a balanced subset of the <code>creditcard.csv</code> dataset by undersampling legitimate transactions.</p>

<ul>
  <li>Uses 30 features including <code>Time</code>, <code>Amount</code>, and <code>V1</code> through <code>V28</code>.</li>
  <li>Accuracy metrics for both train and test sets are printed internally in the model code.</li>
</ul>

<h2>💡 How It Works</h2>

<ol>
  <li>The user is prompted to input 30 feature values.</li>
  <li>When the user clicks "Submit", the model predicts if the transaction is <strong>legitimate</strong> or <strong>fraudulent</strong>.</li>
</ol>

<h2>📦 Requirements</h2>

<pre><code>numpy
pandas
scikit-learn
streamlit</code></pre>

<h2>📁 Dataset Source</h2>
<p>The dataset used is publicly available on <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data" target="_blank">Kaggle - Credit Card Fraud Detection</a>.</p>

<h2>📄 License</h2>
  <p>This project is open-source and available under the <a href="https://github.com/gunavardhanlomada/Deceit-Detector/blob/main/LICENSE" target="_blank">MIT License</a>.</p>

