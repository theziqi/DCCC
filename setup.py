from setuptools import setup, find_packages


setup(name='DCCC',
      version='1.0.0',
      description='Dynamic Clustering and Cluster Contrastive Learning for Unsupervised Person Re-ID with Feature Distribution Alignment',
      author='Ziqi He',
      author_email='heziqi@bupt.edu.cn',
      url='https://github.com/theziqi/DCCC',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])
