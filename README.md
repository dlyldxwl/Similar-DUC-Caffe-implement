# Similar-DUC-Caffe-implement
DUC/STDN similar caffe implementation 

Usage:
```
layer{ 
  name: "enlarge"
  type: "Enlarge"
  bottom: "conv7_2" //size is 5*5
  top: "conv6_2" //size is 10*10
  enlarge_param {
    size: 10  //size after enlarge
  }
}
```
Note:

If the number of bottom channels entered does not divisible by c^2, I did a process similar to the mean pooling.Which should be different from STDN/DUC.
