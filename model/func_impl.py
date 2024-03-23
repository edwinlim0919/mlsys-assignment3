import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    is_fc1: bool,
    is_megatron_mp: bool,
    in_dim: int,
    out_dim: int,
):
    """The function that prepare necessary information for parallel training.

    Parameters
    ----------
        comm : Communicator
            the global mpi communicator

        rank : int
            the corresponding rank of the process

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        is_fc1 : int
            A boolean indicating whether the current layer is the first layer or not

        is_megatron_mp : boolean
            A boolean indicating whether we are using Megatron-style Model Parallel or not

        in_dim : int
            An integer corresponds to the original input feature dimension

        out_dim : int
            An integer corresponds to the original output feature dimension

    Returns
    -------
        mp_idx : int
            An integer corresponds to model parallel communication index

        dp_idx : int
            An integer corresponds to data parallel communication index

        mp_comm : Communicator
            The Model Parallel communicator after split

        dp_comm : Communicator
            The Data Parallel communicator after split

        part_in_dim : int
            An integer corresponds to the input feature dimension after specific parallelism

        part_out_dim : int
            An integer corresponds to the output feature dimension after specific parallelism
    """

    """TODO: Your code here"""

    # Get the mp_idx, dp_idx from rank, mp_size and dp_size (you may not need to use all three of them)
    mp_idx = rank % mp_size
    dp_idx = rank // mp_size

    # Get the model/data parallel communication groups
    # the model/data parallel communication group is required to apply mpi operations within the scope of the group
    # Hint: try to figure out the relationship between the mp_idx, dp_idx with the mp/dp communication group
    #       and use the comm.Split() function to get the corresponding group.
    dp_comm = comm.Split(mp_idx, rank)
    mp_comm = comm.Split(dp_idx, rank)

    # Derive the part_in_dim and part_out_dim depend on is_fc1 and is_megatron_mp
    if is_megatron_mp:
        if is_fc1:
            part_in_dim = in_dim
            part_out_dim = out_dim // mp_size
        else:
            part_in_dim = in_dim // mp_size
            part_out_dim = out_dim
    else:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's inputs across different nodes with naive model parallelism

    Parameters
    ----------
        x : np.ndarray
            layer input for a single node of shape (batch_size, part_in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_x : np.ndarray
            collected layer inputs across different nodes of shape (batch_size, in_dim)

    """

    """TODO: Your code here"""

    # Note: you may want to ensure that the source variable and destination variable in your mpi func call should
    #       have the same data type, otherwise you will not collect the correct value.

    # Hint: Try to figure out the way MPI calls deal with the destination memory layout for 2d matrix transfer, this might
    #       might not align with your expected layout. In order to get the correct layout, you may wish to use some NumPy
    #       functions (np.split and np.concatenate might be helpful).


    recvbuf = np.empty((x.size, mp_size), dtype=np.float64)
    mp_comm.barrier()
    mp_comm.Allgather(x, recvbuf)
    mp_comm.barrier()

    if len(x) < 2:
        result = np.concatenate(recvbuf, axis=0)
        reshaped_result = result.reshape((1, x.size * mp_size))
        return reshaped_result

    subarray_size = mp_size * x.shape[0]
    result_bufs = np.empty((x.size // x.shape[0], mp_size * x.shape[0]), dtype=np.float64)

    for i in range(len(recvbuf)):
        curr_arr = recvbuf[i]
        curr_arr_split = np.array_split(curr_arr, x.shape[0], axis=0)
        start_col = i * x.shape[1]
        end_col = (i+1) * x.shape[1]
        for j in range(len(curr_arr_split)):
            result_bufs[j, start_col:end_col] = curr_arr_split[j]

    return result_bufs


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's outputs across different nodes with naive model parallelism

    Parameters
    ----------
        out : np.ndarray
            layer output for a single node of shape (batch_size, part_out_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_out : np.ndarray
            collected layer outputs across different nodes of shape (batch_size, out_dim)

    """

    """TODO: Your code here"""

    # Hint: you might have just implemented something similar ^-^
    recvbuf = np.empty((out.size, mp_size), dtype=np.float64)
    mp_comm.barrier()
    mp_comm.Allgather(out, recvbuf)
    mp_comm.barrier()

    if len(out) < 2:
        result = np.concatenate(recvbuf, axis=0)
        reshaped_result = result.reshape((1, out.size * mp_size))
        return reshaped_result

    subarray_size = mp_size * out.shape[0]
    result_bufs = np.empty((out.size // out.shape[0], mp_size * out.shape[0]), dtype=np.float64)

    for i in range(len(recvbuf)):
        curr_arr = recvbuf[i]
        curr_arr_split = np.array_split(curr_arr, out.shape[0], axis=0)
        start_col = i * out.shape[1]
        end_col = (i+1) * out.shape[1]
        for j in range(len(curr_arr_split)):
            result_bufs[j, start_col:end_col] = curr_arr_split[j]

    return result_bufs

    #raise NotImplementedError


def megatron_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's inputs across different nodes with megatron-style model parallelism

    Parameters
    ----------
        x : np.ndarray
            layer input for a single node of shape (batch_size, part_in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_x : np.ndarray
            collected layer inputs across different nodes of shape (batch_size, in_dim)

    """

    """TODO: Your code here"""

    # Hint: you don't need all the input parameters to get the collected_x

    if len(x) < 2:
        return x.reshape((1, x.size))
    return x


def megatron_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's outputs across different nodes with megatron-style model parallelism

    Parameters
    ----------
        out : np.ndarray
            layer output for a single node of shape (batch_size, part_out_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_out : np.ndarray
            collected layer outputs across different nodes of shape (batch_size, out_dim)

    """

    """TODO: Your code here"""

    # Hint: try to work through a toy forward example for megatron-style model parallel to figure out the
    #       the communication functions that you might need


    #recvbuf = np.empty((out.size, mp_size), dtype=np.float64)
    recvbuf = np.empty(out.size, dtype=np.float64)
    mp_comm.barrier()
    #mp_comm.Allgather(out, recvbuf)
    mp_comm.Allreduce(out, recvbuf, op=MPI.SUM)
    mp_comm.barrier()

    print(f'out: {out}')
    print(f'out.shape: {out.shape}')
    print(f'recvbuf: {recvbuf}')
    print(f'mp_comm: {mp_comm}')
    print(f'mp_size: {mp_size}')

    return recvbuf.reshape((1, recvbuf.size))


def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """The function for collecting layer fc2's output_grad across different nodes with naive model parallelism

    Parameters
    ----------
        output_grad : np.ndarray
            layer output_grad for a single node of shape (batch_size, out_dim)

        mp_group_idx : int
            The Model Parallel group idx

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_output_grad : np.ndarray
            collected layer output_grad across different nodes of shape (batch_size, part_out_dim)

    """

    """TODO: Your code here"""

    # Hint: you might want to use np.split to get the collected_output_grad for each MP node

    raise NotImplementedError


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's grad_x across different nodes with naive model parallelism

    Parameters
    ----------
        grad_x : np.ndarray
            layer backward grad_x for a single node of shape (batch_size, in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_grad_x : np.ndarray
            collected layer backward grad_x across different nodes of shape (batch_size, part_in_dim)

    """

    """TODO: Your code here"""

    # Hint 1: The communication pattern for this function can be seen as the reverse of its forward
    #         , so you might to check the naive_collect_forward_output() impl.

    # Hint 2: You might want to use reduce_scatter

    raise NotImplementedError


def megatron_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """The function for collecting layer fc2's output_grad across different nodes with megatron-style model parallelism

    Parameters
    ----------
        output_grad : np.ndarray
            layer output_grad for a single node of shape (batch_size, out_dim)

        mp_group_idx : int
            The Model Parallel group idx

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_output_grad : np.ndarray
            collected layer output_grad across different nodes of shape (batch_size, part_out_dim)

    """

    """TODO: Your code here"""

    # Hint: your implementation should be within one line of code

    raise NotImplementedError


def megatron_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's grad_x across different nodes with megatron-style model parallelism

    Parameters
    ----------
        grad_x : np.ndarray
            layer backward grad_x for a single node of shape (batch_size, in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_grad_x : np.ndarray
            collected layer backward grad_x across different nodes of shape (batch_size, part_in_dim)

    """

    """TODO: Your code here"""

    # Hint: your implementation should be within one line of code

    raise NotImplementedError


def collect_weight_grad(
    grad_w: np.ndarray,
    grad_b: np.ndarray,
    dp_comm,
):
    """The function for collecting weight gradients across data parallel nodes

    Parameters
    ----------
        grad_w : np.ndarray
            gradients value for fc weight on a single node of shape (in_dim, out_dim)

        grad_b : np.ndarray
            gradients value for fc bias on a single node of shape (1, out_dim)

        dp_comm : Communicator
            The Data Parallel communicator

    Returns
    -------
        collected_grad_w : np.ndarray
            collected gradients value of shape (in_dim, out_dim) for fc weight across different nodes

        collected_grad_b : np.ndarray
            collected gradients value of shape (1, out_dim) for fc bias across different nodes

    """

    """TODO: Your code here"""

    # Hint: Think about how you might want to aggregate the gradients from different nodes in data parallel training

    raise NotImplementedError
