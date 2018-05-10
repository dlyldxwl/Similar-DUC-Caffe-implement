#include <algorithm>
#include <vector>
#include "caffe/layers/enlarge_layer.hpp"

namespace caffe {

template <typename Dtype>
void EnlargeLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    img_size_ = this->layer_param_.enlarge_param().size();
    CHECK_GT(img_size_,0)<<"feature map size must be greater than 0";

    ch_ori_ = bottom[0]->channels();
    h_ori_ = bottom[0]->height();
    w_ori_ = bottom[0]->width();

    scale_ = int(img_size_ / h_ori_);
    group_ = int(ch_ori_ / (scale_*scale_)); //channels after enlarge
	const int a = int(img_size_%h_ori_);

	CHECK_EQ(a,0)<<"The size parameter must be a multiple of the bottom feature map height/width ";
    CHECK_EQ(h_ori_,w_ori_)<<"the width and height of the feature map to be sampled are equal";
    CHECK_GT(img_size_,h_ori_)<<"size param need be greater than feature map size";
    
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    img_size_ = this->layer_param_.enlarge_param().size();
    batch_ = bottom[0]->num();
    ch_ori_ = bottom[0]->channels();
    h_ori_ = bottom[0]->height();
    w_ori_ = bottom[0]->width();

    scale_ = int(img_size_ / h_ori_);
    group_ = int(ch_ori_ / (scale_*scale_));

    top[0]->Reshape(batch_,group_,img_size_,img_size_);
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	const int sp_os = bottom[0]->count(2);
	const int sp_ns = top[0]->count(2); 
	const int extra_maps_ = int(ch_ori_%(scale_*scale_));

	for (int m = 0; m < batch_; ++m)
	{
		for (int n = 0; n < group_; ++n)
		{
			if ((n!=group_-1)||(extra_maps_==0)) 
			{
				for (int h = 0; h < img_size_; ++h)
				{
					for (int w = 0; w < img_size_; ++w)
					{
						int index_n = h*img_size_ + w; //index of top(new) feature map
						int index_o = (h%scale_*scale_ + w%scale_)*sp_os + (h / scale_*w_ori_ + w / scale_);// index of bottom(old) feature map
						top_data[index_n] = bottom_data[index_o];
					}
				}
				bottom_data += scale_*scale_*sp_os;
				top_data += sp_ns;
			}
			else
			{
				for (int h = 0; h < img_size_; ++h)
				{
					for (int w = 0; w < img_size_; ++w)
					{
						int index_n = h*img_size_ + w; //index of top(new) feature map
						int map_ind_o = h%scale_*scale_ + w%scale_;
						if (map_ind_o != scale_*scale_-1) 
						{
							int index_o = map_ind_o*sp_os + (h / scale_*w_ori_ + w / scale_);// index of bottom(old) feature map
							top_data[index_n] = bottom_data[index_o];
						}
						else
						{
							Dtype sum=0.0;
							for (int i=0;i<=extra_maps_;++i)
							{
								int index_extra = (map_ind_o+i)*sp_os + (h / scale_*w_ori_ + w / scale_);
								sum+=bottom_data[index_extra];
							}
							Dtype ave=sum/(extra_maps_+1);
							top_data[index_n] = ave;
						}
					}
				}
			}
		}
	}	
}

template <typename Dtype>
void EnlargeLayer<Dtype>:: Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if(propagate_down[0])
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const int sp_ns = top[0]->count(2);
		const int sp_os = bottom[0]->count(2);

		const int extra_maps_ = int(ch_ori_%(scale_*scale_));

		for (int m = 0; m < batch_; ++m)
		{
			for (int n = 0; n < group_; ++n)
			{
				if ((n!=group_-1)||(extra_maps_==0))
				{
					for (int h = 0; h < img_size_; ++h)
					{
						for (int w = 0; w < img_size_; ++w)
						{
							int index_n = h*img_size_ + w; //index of top feature map
							int index_o = (h%scale_*scale_ + w%scale_)*sp_os + (h / scale_*w_ori_ + w / scale_);// index of bottom feature map
							bottom_diff[index_o] = top_diff[index_n];
						}
					}
					bottom_diff += scale_*scale_*sp_os;
					top_diff += sp_ns;
				}
				else
				{
					for (int h = 0; h < img_size_; ++h)
					{
						for (int w = 0; w < img_size_; ++w)
						{
							int index_n = h*img_size_ + w; //index of top(new) feature map
							int map_ind_o = h%scale_*scale_ + w%scale_;
							if (map_ind_o != scale_*scale_-1)
							{
								int index_o = map_ind_o*sp_os + (h / scale_*w_ori_ + w / scale_);
								bottom_diff[index_o] = top_diff[index_n];
							}
							else
							{
								Dtype ave_diff = top_diff[index_n]/(extra_maps_+1);
								for (int i=0;i<=extra_maps_;++i)
								{
									int index_extra = (map_ind_o+i)*sp_os + (h / scale_*w_ori_ + w / scale_);
									bottom_diff[index_extra] = ave_diff;
								}
							} 
							
						}
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(EnlargeLayer);
#endif

INSTANTIATE_CLASS(EnlargeLayer);
REGISTER_LAYER_CLASS(Enlarge);
	
}
