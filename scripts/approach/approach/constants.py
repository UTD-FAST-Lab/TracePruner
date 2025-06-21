
IN_COLAB = False

if IN_COLAB:
    base = '/content/drive/MyDrive/Research/Program-analysis/Projects/project1/data/xcorp/results'
    static_cg_dir = '/content/drive/MyDrive/Research/Program-analysis/Projects/project1/data/xcorp/dataset'
    features_dir = '/content/drive/MyDrive/Research/Program-analysis/Projects/project1/data/xcorp/features'
    manual_labels_dir = '/content/drive/MyDrive/Research/Program-analysis/Projects/project1/data/xcorp/manual_labeling'
else:
    base = '/20TB/mohammad/xcorpus-total-recall/results'
    static_cg_dir = '/20TB/mohammad/xcorpus-total-recall/dataset'
    features_dir = '/20TB/mohammad/xcorpus-total-recall/features'
    manual_labels_dir = '/20TB/mohammad/xcorpus-total-recall/manual_labeling'