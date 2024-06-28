import tensorflow as tf


def calc_spline_values(x: tf.Tensor, grid: tf.Tensor, spline_order: int):
    """
    Calculate B-spline values for the input tensor

    Parameters
    - `x: tf.Tensor` Input tensor with shape `(batch_size, in_size)`
    - `grid: tf.Tensor` Grid tensor with shape `(in_size, grid_size + 2 * spline_order + 1)`
    - `spline_order: int` Spline order

    Returns: `tf.Tensor` B-spline bases tensor with shape (batch_size, in_size, grid_size + spline_order)
    """

    # Il tensore in input deve essere di rango 2 (matrice 2D) | Dimensione = (batch_size, n_records)
    assert x.shape.rank == 2
    print(tf.shape(x))
    
    # Aggiunta di una dimensione sull'ultimo asse | Dimensione = (batch_size, n_records, 1)
    x = tf.expand_dims(x, axis=-1)

    # Definizione della B-spline di grado 0
    bases = tf.logical_and(
        tf.greater_equal(x, grid[:, :-1]), tf.less(x, grid[:, 1:])
    )
    bases = tf.cast(bases, x.dtype)
    
    # Definizione ricorsiva delle B-spline dei gradi da 1 a spline_order
    for k in range(1, spline_order+1):
        bases = (
            (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
        ) + (
            (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:]
        )

    return bases


def fit_spline_coef(x: tf.Tensor, y: tf.Tensor, grid: tf.Tensor, spline_order: int, l2_reg: float = 0, fast: bool = True):
    """
    Fit the spline coefficients for given spline input and spline output tensors,\n
    the formula is spline output `y_{i,j} = sum_{k=1}^{grid_size + spline_order} coef_{i,j,k} * B_{k}(x_i)`\n
    in which, `i=1:in_size, j=1:out_size`. written in matrix form, `Y = B @ coef`,\n
    - `Y` with shape `(batch_size, in_size, out_size)`
    - `B` is the B-spline bases tensor `B_{k}(x_i)` with shape `(batch_size, in_size, grid_size + spline_order)`
    - `coef` is the spline coefficients tensor with shape `(in_size, grid_size + spline_order, out_size)`

    `in_size` is a independent dimension, `coef` transform the `grid_size + spline_order` to `out_size`

    Parameters:
    - `x: tf.Tensor` Spline input tensor with shape `(batch_size, in_size)`
    - `y: tf.Tensor` Spline output tensor with shape `(batch_size, in_size, out_size)`
    - `grid: tf.Tensor` Spline grid tensor with shape `(in_size, grid_size + 2 * spline_order + 1)`
    - `spline_order: int` Spline order
    - `l2_reg: float` L2 regularization factor for the least square solver
    - `fast: bool` Whether to use the fast solver for the least square problem
    
    Returns: `tf.Tensor` Spline coefficients tensor with shape `(in_size, grid_size + spline_order, out_size)`
    """

    # Calcolo delle B-spline B_{k}(x_i) (senza coefficienti) | Dimensione = (batch_size, n_records, grid_size + spline_order) RICONTROLLARE!
    B = calc_spline_values(x, grid, spline_order)
    B = tf.transpose(B, perm=[1, 0, 2]) # Reshaping | Dimensione = (n_records, batch_size, grid_size + spline_order)

    # Reshaping del vettore target | Dimensione = (batch_size, n_records, out_size) -> (n_records, batch_size, out_size)
    y = tf.transpose(y, perm=[1, 0, 2])

    # Risoluzione dell'equazione lineare per l'assegnazione dei coefficienti alle B-spline in modo che fittino con y
    # Dimensione = (n_records, grid_size + spline_order, out_size) Spiegazione: l'i-esimo coefficiente ha dimensione pari a quella di ogni B-spline
    coef = tf.linalg.lstsq(B, y, l2_regularizer=l2_reg, fast=fast)

    return coef