# Similar-DUC-Caffe-implement
DUC/SYDN similar caffe implementation 

Usage:

layer{
  name: "enlarge"
  type: "Enlarge"
  bottom: "conv7_2"
  top: "conv6_2"
  enlarge_param {
    size: 10 //size after enlarge
  } 
}

Note:

If the number of bottom channels entered does not divisible by c^2, I did a process similar to the mean pooling.Which should be different from STDN/DUC.
