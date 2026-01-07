import numpy as np

def spike_removal_p(original_signal: np.ndarray, fs: float) -> np.ndarray:
    """Python implementation of schmidt spike removal"""
    initial_shape = original_signal.shape

    # Find the window size (500 ms)
    windowsize = round(float(fs) / 2.0)

    # Find any samples outside of an integer number of windows
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows
    sampleframes = original_signal[:len(original_signal) - trailingsamples].reshape((windowsize, -1), order='F')

    # Find the Maximum Absolute Amplitudes (MAAs)
    MAAs = np.max(np.abs(sampleframes), axis=0)

    max_iterations = 1000  
    iteration = 0

    while np.any(MAAs > np.median(MAAs) * 3) and iteration < max_iterations:
        previous_MAAs = MAAs.copy()
        window_num = np.argmax(MAAs)
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))

        zero_crossings = np.concatenate([(np.abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1), np.zeros(1)])

        spike_start = 0 if not np.any(zero_crossings[:spike_position]) else np.where(zero_crossings[:spike_position])[0][-1] + 1
        zero_crossings[:spike_position] = 0
        spike_end = windowsize - 1 if not np.any(zero_crossings) else np.where(zero_crossings)[0][0]

        sampleframes[spike_start:spike_end, window_num] = 0.0001

        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

        # Stop if no change in MAAs (convergence)
        if np.array_equal(MAAs, previous_MAAs):
            break
        
        iteration += 1

    # Reshape the despiked signal back to its original form
    despiked_signal = sampleframes.flatten(order='F')

    # Add trailing samples back
    despiked_signal = np.concatenate((despiked_signal, original_signal[len(despiked_signal):]))

    assert despiked_signal.shape == initial_shape, "Shape should remain unchanged" 

    return despiked_signal