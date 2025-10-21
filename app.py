import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
segment1 = np.random.normal(loc=30, scale=5, size=100)
segment2 = np.random.normal(loc=60, scale=10, size=150)
segment3 = np.random.normal(loc=90, scale=8, size=120)
data = np.concatenate([segment1, segment2, segment3]).reshape(-1, 1)


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(data_scaled)
clusters = gmm.predict(data_scaled)
means = scaler.inverse_transform(gmm.means_).flatten()


st.set_page_config(page_title="Customer Segmentation (GMM)", page_icon="💰", layout="centered")

st.title("💰 Customer Segmentation using Gaussian Mixture Model")
st.write("""
This app segments customers based on their **spending behavior** using a **Gaussian Mixture Model (GMM)**.
""")


st.subheader("📊 Cluster Centers (Approx. Spendings):")
for i, m in enumerate(means):
    st.write(f"**Cluster {i}** → ₹{m:.2f}")


st.subheader("🔍 Predict Customer Segment")

spending = st.number_input("Enter customer spending (₹):", min_value=0.0, max_value=200.0, value=50.0)
if st.button("Predict Segment"):
    user_input = np.array([[spending]])
    user_input_scaled = scaler.transform(user_input)
    prediction = gmm.predict(user_input_scaled)[0]

   
    segment_labels = {0: "Low Spender", 1: "Mid Spender", 2: "High Spender"}
    label = segment_labels[prediction]

    st.success(f"Predicted Segment: **Cluster {prediction} ({label})**")


st.subheader("📈 Cluster Visualization")

fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(data, np.zeros_like(data), c=clusters, cmap="Accent", alpha=0.6)
ax.scatter(means, np.zeros_like(means), c='red', marker='x', s=100, label='Cluster Centers')
ax.set_xlabel("Purchase Amount (₹)")
ax.set_title("Customer Segmentation (GMM)")
ax.legend()
st.pyplot(fig)


st.subheader("📉 Cluster Probability Curves")

x = np.linspace(0, 120, 500).reshape(-1, 1)
x_scaled = scaler.transform(x)
probs = gmm.predict_proba(x_scaled)

fig2, ax2 = plt.subplots()
ax2.plot(x, probs)
ax2.set_xlabel("Purchase Amount (₹)")
ax2.set_ylabel("Cluster Probability")
ax2.legend(["Cluster 0", "Cluster 1", "Cluster 2"])
st.pyplot(fig2)
