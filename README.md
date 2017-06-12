# MXnet Workshop

This repository contains the code used in the AWS Summit workshop: Deep learning at Cloud Scale and AI as a service, held in Washington D.C. on June 12, 2017.

The workshops consists of slides, where are found here: [AWS_Summit_Workshop_Intel.pdf](https://s3-us-west-1.amazonaws.com/nervana-course/mxnet_workshop/AWS_Summit_Workshop_Intel.pdf), and three jupyter notebooks using the code in this repository.

To launch the notebooks on AWS, use the following settings (otherwise default settings):
1. AMI: `ami-6ef6d40e` (us-west-1) This AMI is based on the [Amazon Deep Learning AMI](https://aws.amazon.com/amazon-ai/amis/).
2. Instance type: `c4.8xlarge`
3. Security group: Use Custom TCP Rule, and allow port 8888. For example, see below:

<img src=https://s3-us-west-1.amazonaws.com/nervana-course/mxnet_workshop/notebook_security_group.png>

After launching, the notebooks will be accessible on: `http://xxx.xx.xx.xxx:8888`. The password is: `deeplearning`.

