import numpy as np

def exact_solution(Rnum, t, x):
    t0 = np.exp(Rnum / 8.0)
    return (x / (t + 1)) / (1.0 + np.sqrt((t + 1) / t0) * np.exp(Rnum * (x * x) / (4.0 * t + 4)))


def collect_snapshots(Rnum, x , tsteps):
    snapshot_matrix = np.zeros(shape=(np.shape(x)[0], np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix[:, t] = exact_solution(Rnum, tsteps[t], x)[:]

    return snapshot_matrix


def collect_multiparam_snapshots_train(x, tsteps):
    rnum_vals = np.arange(100, 2000, 100)

    rsnap = 0
    for rnum_val in rnum_vals:
        snapshots_temp = np.transpose(collect_snapshots(rnum_val, x , tsteps))

        if rsnap == 0:
            all_snapshots = snapshots_temp
        else:

            all_snapshots = np.concatenate((all_snapshots, snapshots_temp), axis=0)

        rsnap = rsnap + 1
    return all_snapshots, rnum_vals / 1000


def collect_multiparam_snapshots_test(x, tsteps):
    rnum_vals = np.arange(50, 2500, 200)

    rsnap = 0
    for rnum_val in rnum_vals:
        snapshots_temp = np.transpose(collect_snapshots(rnum_val, x , tsteps))

        if rsnap == 0:
            all_snapshots = snapshots_temp
        else:

            all_snapshots = np.concatenate((all_snapshots, snapshots_temp), axis=0)

        rsnap = rsnap + 1
    return all_snapshots, rnum_vals / 1000