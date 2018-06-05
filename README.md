# Similar-DUC-Caffe-implement
DUC/STDN similar caffe implementation 

Usage:

layer{ 
>name: "enlarge"<br>
  type: "Enlarge"<br>
  bottom: "conv7_2"<br>
  top: "conv6_2"<br>
  enlarge_param {
  >>size: 10  //size after enlarge<br>
  }<br>
}<br>

Note:

If the number of bottom channels entered does not divisible by c^2, I did a process similar to the mean pooling.Which should be different from STDN/DUC.
