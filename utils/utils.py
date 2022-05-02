def my_kendall_tau(batch_perm1, batch_perm2):
    """Wraps scipy.stats kendalltau function.

    Args:
      batch_perm1: A 2D tensor (a batch of matrices) with
        shape = [batch_size, N]
      batch_perm2: same as batch_perm1

    Returns:
      A list of Kendall distances between each of the elements of the batch.
    """

    def kendalltau_batch(x, y):

        if x.ndim == 1:
            x = np.reshape(x, [1, x.shape[0]])
        if y.ndim == 1:
            y = np.reshape(y, [1, y.shape[0]])
        kendall = np.zeros((x.shape[0], 1), dtype=np.float32)
        for i in range(x.shape[0]):
            kendall[i, :] = kendalltau(x[i, :], y[i, :])[0]
        return kendall

    listkendall = kendalltau_batch(batch_perm1.cpu().numpy(), batch_perm2.cpu().numpy())
    listkendall = torch.from_numpy(listkendall)
    return listkendall
