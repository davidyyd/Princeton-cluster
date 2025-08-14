# New Manual

Please check out [Taiming Lu's guide](https://github.com/TaiMingLu/Della-Manual) on Della and [my guide](https://github.com/davidyyd/Neuronic-Manual) on Neuronic. These guides are more comprehensive and include more information on the cluster setup and usage.

# Cluster Manual [Deprecated]

## Overview
This repo is a simple guide to help you connect to the Princeton cluster and launch your first PyTorch training script. 

We provide two different versions: [CAMPUS.md](CAMPUS.md) for campus-wide clusters (e.g., adroit, della) and [DEPARTMENT.md](DEPARTMENT.md) for department-wide clusters (e.g., neuronic). Most of their contents overlaps, with only minor differences in SSH configuration and environment setup.

## Account Setup

To connect to any cluster, you should have either a Princeton account or Research Computer User (RCU) account (for external collaborators). The account should have been granted access to the cluster. Your account should also be configured with two-factor authentication. Follow [this link](https://princeton.edu/duoportal) to set up two-factor authentication. 

It is important to connect to the campus network when accessing the cluster. If you are off campus, use GlobalProtect VPN to connect to the campus network. See the detailed instructions [here](https://princeton.service-now.com/service?sys_id=KB0012373&id=kb_article) to set up this. This is not needed when the computer has been directly connected to the campus wifi (eduroam). 

## Resources

Cluster Pages:
1. Adroit: https://researchcomputing.princeton.edu/systems/adroit
2. Della: https://researchcomputing.princeton.edu/systems/della
3. Neuronic: https://clusters.cs.princeton.edu/
4. Ionic: https://csguide.cs.princeton.edu/resources/clusters


More Resources:
1. Slurm: https://researchcomputing.princeton.edu/support/knowledge-base/slurm
2. PyTorch: https://researchcomputing.princeton.edu/support/knowledge-base/pytorch
3. Huggingface: https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face
4. VSCode: https://researchcomputing.princeton.edu/support/knowledge-base/vs-code
5. Sharing Data: https://researchcomputing.princeton.edu/support/knowledge-base/sharing-data


## Acknowledgement

This guide is adapted from [here](https://github.com/PrincetonUniversity/multi_gpu_training). Their repository is more comprehensive and includes examples on using PyTorch Lightning, FSDP, and TensorFlow on the cluster.

## Contact

If you have any questions in general about the setup or usage of the cluster, please contact David Yin by [email](yida.yin@princeton.edu) or [messenger](https://www.facebook.com/yida.yin.5?mibextid=wwXIfr&mibextid=wwXIfr).