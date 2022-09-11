<h1> <b> Viral Epitope Class I MHC Prediction Project </b></h1>
<i> currently in progress </i>
<h2> Goals </h2>
<p> To design a deep learning model that will accurately predict the binding of pan viral epitopes to class I MHC modules. We will use the benchmark dataset retrieved from the Immune Epitope Database (IEDB) to train and validate our models. </p>
<p> Both peptide and MHC sequences will be transformed into a pseudosequences for representation. They will be further transformed using a BLOSUM50 matrix to represent amino acid composition. As we are focusing on only class I MHC presentation, the viral epitopes will only consist of sequence lengths from 9 peptides to 15 peptides. The MHC pseudosequence will be represented only by the known anchor points, amino acids where the peptides bind to for antigen presentation. </p>
<b> Current status as of September 2022: </b>
<p> We have developed a 1-layer CNN with self-attention to predict the viral epitopes. The performance, as measured by AUC, for the Receiver-Operating Characteristic and Precision-Recall Curve range from 0.88 to 0.90. While this looks promising, we will be benchmarking the performance using several machine learning models. The ML models we will use are: </p>
<ol>
<li> Decision Tree 
<li> Random Forest
<li> SVM
<li> K Nearest Neighbors
<li> Logistic Regression
</ol>
We will also test several other deep learning models. The deep learning structures we are interested in testing are (and the list is not conclusive yet):
<ol>
<li> CNN + self-attention
<li> LSTM
<li> Transformer
<li> More to be determined
</ol>
Furthermore, to improve performance, we will utilize a balanced training method by either generating an ensemble model or upsampling the positive data. 
