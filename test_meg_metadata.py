import numpy as np

meta = np.load('data/things-eeg2/stimuli/image_metadata.npy', allow_pickle=True).item()

print(list(meta.keys()))

# The MEG events have 'value' up to 16540 ? Or is it the concept? Let's check a few
print("Test image concepts array length:", len(meta['test_img_concepts_THINGS']))
print("First five test concepts:", meta['test_img_concepts_THINGS'][:5])
print("First five train concepts:", meta['train_img_concepts_THINGS'][:5])
