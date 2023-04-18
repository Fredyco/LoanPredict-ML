import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="Hea",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run():
    # Membuat Tittle
    st.title("Loan Status Prediction")

    # Membuat Sub Header
    st.subheader("EDA untuk Analisa Dataset Heart Failure")

    # Menambahkan Gambar
    image = Image.open('mls2.png')
    st.image(image)

    st.markdown("---")

    df = pd.read_csv("loan_predict.csv")
    df = df.dropna()
    st.dataframe(df)

    # Eksplorasi Target
    st.write("### Eksplorasi Kolom Target")
    target = df["Loan_Status"].value_counts().reset_index()
    persen = df["Loan_Status"].value_counts(normalize=True).reset_index()
    target["percentage"] = persen["Loan_Status"]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(target["percentage"], labels = target["index"], autopct='%.0f%%')
    ax.set_title('Pie Chart Loan Status')
    ax.legend(fontsize=12)
    st.pyplot(fig)
    st.dataframe(target)
    st.write('Dari tabel dan plot diatas didapatkan bahwa data dependen atau kolom target imbalanced, dimana pengajuan yang diapprove sebanyak 69% dan pengajuan yang tidak di approve sebanyak 31%.')

    # Eksplorasi Data Numeric
    st.write("### Eksplorasi Data Numeric")

    # Applicant Income
    st.write('#### ApplicantIncome')
    no_failure_cp = df[df["Loan_Status"] == 'N']["ApplicantIncome"].reset_index()
    failure_cp = df[df["Loan_Status"] == 'Y']["ApplicantIncome"].reset_index()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    axes[0].hist(no_failure_cp["ApplicantIncome"])
    axes[0].set_title("Applicant Income Not Approved")
    axes[1].hist(failure_cp["ApplicantIncome"], color='orange')
    axes[1].set_title("Applicant Income Approved")
    st.pyplot(fig)
    st.write('Nilai maksimum dari Applicant Income orang yang tidak di approve adalah :  81000')
    st.write('Nilai minimum dari Applicant Income orang yang tidak di approve adalah :  150')
    st.write('Nilai rata-rata dari Applicant Income orang yang tidak di approve adalah :  5730.189189189189')
    st.write('')
    st.write('Nilai maksimum dari Applicant Income orang yang di approve adalah :  39999')
    st.write('Nilai minimum dari Applicant Income orang yang di approve adalah :  645')
    st.write('Nilai rata-rata dari Applicant Income orang yang di approve adalah :  5201.093373493976')
    st.write('')
    st.write('Dari tabel dan plot diatas ditemukan bahwa rata-rata applicant income tidak terlalu berbeda antara pinjaman yang di approve dan tidak. Akan tetapi nilai minimal Applicant income untuk pengajuan pinjaman adalah 645.')

    # Co Applicant Income
    st.write('#### CoapplicantIncome')
    no_failure_cp = df[df["Loan_Status"] == 'N']["CoapplicantIncome"].reset_index()
    failure_cp = df[df["Loan_Status"] == 'Y']["CoapplicantIncome"].reset_index()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    axes[0].hist(no_failure_cp["CoapplicantIncome"])
    axes[0].set_title("Co applicant Income Not Approved")
    axes[1].hist(failure_cp["CoapplicantIncome"], color='orange')
    axes[1].set_title("Co applicant Income Approved")
    st.pyplot(fig)
    st.write('Nilai maksimum dari Co applicant Income orang yang tidak di approve adalah :  33837.0')
    st.write('Nilai minimum dari Co applicant Income orang yang tidak di approve adalah :  0.0')
    st.write('Nilai rata-rata dari Co applicant Income orang yang tidak di approve adalah :  1773.081081081081')
    st.write('')
    st.write('Nilai maksimum dari Co applicant Income orang yang di approve adalah :  20000.0')
    st.write('Nilai minimum dari Co applicant Income orang yang di approve adalah :  0.0')
    st.write('Nilai rata-rata dari Co applicant Income orang yang di approve adalah :  1495.508795146506')
    st.write('')
    st.write('Dari tabel dan plot diatas ditemukan bahwa rata-rata Co applicant income tidak terlalu berbeda antara pinjaman yang di approve dan tidak. Bahkan rata-rata nilai Co Applicant Income dari pinjaman yang di approve itu lebih kecil dibanding rata-rata Co Applicant Income pinjaman yang tidak di approve.')

    # Loan Amount
    st.write('#### LoanAmount')
    no_failure_cp = df[df["Loan_Status"] == 'N']["LoanAmount"].reset_index()
    failure_cp = df[df["Loan_Status"] == 'Y']["LoanAmount"].reset_index()

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    axes[0].hist(no_failure_cp["LoanAmount"])
    axes[0].set_title("Loan Amount Not Approved")
    axes[1].hist(failure_cp["LoanAmount"], color='orange')
    axes[1].set_title("Loan Amount Approved")
    st.pyplot(fig)
    st.write('Nilai maksimum dari Loan Amount orang yang tidak di approve adalah :  570.0')
    st.write('Nilai minimum dari Loan Amount orang yang tidak di approve adalah :  9.0')
    st.write('Nilai rata-rata dari Loan Amount orang yang tidak di approve adalah :  153.3783783783784')
    st.write('')
    st.write('Nilai maksimum dari Loan Amount orang yang di approve adalah :  600.0')
    st.write('Nilai minimum dari Loan Amount orang yang di approve adalah :  17.0')
    st.write('Nilai rata-rata dari Loan Amount orang yang di approve adalah :  140.88253012048193')
    st.write('')
    st.write('Dari tabel dan plot diatas didapatkan bahwa tidak ada perbedaan yang sangat signifikan terhadap loan amount dari pinjaman yang di approve dan yang tidak. Akan tetapi rata-rata pinjaman yang approve itu sedikit lebih rendah dari rata-rata pinjaman yang tidak di approve.')

    # Eksplorasi data Sex
    st.write("### Eksplorasi Kolom Categorical Nominal")

    st.write("#### Gender")
    sex_total = df["Gender"].value_counts().reset_index()
    persen_sex = df["Gender"].value_counts(normalize=True).reset_index()
    sex_total["percentage"] = persen_sex["Gender"]

    sex = df.groupby(["Gender"], as_index=False)["Loan_Status"].value_counts()
    percentage_acc_sex = df.groupby(["Gender"], as_index=False)["Loan_Status"].value_counts(normalize=True)
    sex["percentage"] = percentage_acc_sex["proportion"]

    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(sex_total["percentage"], labels=sex_total["index"], autopct='%.0f%%')
    ax.set_title("Jenis Kelamin Pengaju Pinjaman")
    ax.legend(fontsize=12)

    st.pyplot(fig)
    st.dataframe(sex)
    st.write('Dari tabel dan plot diatas didapatkan bahwa mayoritas pengaju pinjaman adalah laki-laki, Selain itu didapatkan juga bahwa pengaju pinjaman berjenis kelamin pria memiliki peluang lebih besar untuk di approve pinjamannya jika dibandingkan dengan perempuan.')

    # Property_Area
    st.write("#### Property_Area")
    sex_total = df["Property_Area"].value_counts().reset_index()
    persen_sex = df["Property_Area"].value_counts(normalize=True).reset_index()
    sex_total["percentage"] = persen_sex["Property_Area"]

    sex = df.groupby(["Property_Area"], as_index=False)["Loan_Status"].value_counts()
    percentage_acc_sex = df.groupby(["Property_Area"], as_index=False)["Loan_Status"].value_counts(normalize=True)
    sex["percentage"] = percentage_acc_sex["proportion"]

    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(sex_total["percentage"], labels=sex_total["index"], autopct='%.0f%%')
    ax.set_title("Persentase Tempat Tinggal Pengaju Pinjaman")
    ax.legend(fontsize=12)

    st.pyplot(fig)
    st.dataframe(sex)
    st.write('Dari tabel dan plot diatas didapatkan bahwa tempat tinggal pengaju pinjaman mempengaruhi apakah pinjaman di approve atau tidak, hal ini dapat dibuktikan dari pengaju pinjaman yang tinggal di ara Semiurban memiliki kemungkinan pinjaman di approve lebih besar dari orang yang tinggal di Rural dan Urban.')

    # Education
    st.write("#### Education")
    sex_total = df["Education"].value_counts().reset_index()
    persen_sex = df["Education"].value_counts(normalize=True).reset_index()
    sex_total["percentage"] = persen_sex["Education"]

    sex = df.groupby(["Education"], as_index=False)["Loan_Status"].value_counts()
    percentage_acc_sex = df.groupby(["Education"], as_index=False)["Loan_Status"].value_counts(normalize=True)
    sex["percentage"] = percentage_acc_sex["proportion"]

    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(sex_total["percentage"], labels=sex_total["index"], autopct='%.0f%%')
    ax.set_title("Klasifikasi Education Pengaju Pinjaman")
    ax.legend(fontsize=12)

    st.pyplot(fig)
    st.dataframe(sex)
    st.write('Dari tabel dan plot diatas didapatkan bahwa mayoritas edukasi pengaju pinjaman itu adalah Graduate, Selain itu didapatkan bahwa orang yang memiliki tingkat pendidikan Graduate itu memiliki peluang lebih besar untuk di approve pinjamannya dibanding orang yang tidak graduate.')

if __name__ == "__app__":
    run()