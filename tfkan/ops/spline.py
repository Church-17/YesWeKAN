import tensorflow as tf

def calc_spline_values(x: tf.Tensor, grid: tf.Tensor, spline_order: int):
    """
    Calcola i valori B-spline per un tensore di input   

    Parameters
    - `x: tf.Tensor` tensore in input con forma `(batch_size, in_size)`
    - `grid: tf.Tensor` tensore griglia di forma `(in_size, grid_size + 2 * spline_order + 1)`
    - `spline_order: int` ordine delle spline

    Returns: `tf.Tensor` ritorna un tensore con una B-spline di forma (batch_size, in_size, grid_size + spline_order)
    """

    # Il tensore in input deve essere di rango 2 (matrice 2D) | Dimensione = (batch_size, n_records), altrimenti si genera un errore
    assert x.shape.rank == 2
    print(tf.shape(x)) #serve???
    
    # Aggiunta di una dimensione sull'ultimo asse | Dimensione = (batch_size, n_records, 1)
    x = tf.expand_dims(x, axis=-1)

    # Definizione della B-spline di grado 0
    bases = tf.logical_and(
        tf.greater_equal(x, grid[:, :-1]), tf.less(x, grid[:, 1:]) #crea un tensore booleano che controlla se x è maggiore del corrispondente elemento di grid[:, :-1] e minore di grid[:, 1:]
    )
    bases = tf.cast(bases, x.dtype) #converte il tensore booleano nel tipo di dati di x
    
    # Definizione ricorsiva delle B-spline dei gradi da 1 a spline_order
    for k in range(1, spline_order+1): #scorre per gli ordini delle spline
        bases = ( #aggiorna le spline di ordine k
            (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
        ) + (
            (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:]
        )

    return bases


def fit_spline_coef(x: tf.Tensor, y: tf.Tensor, grid: tf.Tensor, spline_order: int, l2_reg: float = 0, fast: bool = True):
    """
        Adatta i coefficienti spline per i tensori di input e output spline dati,\n
        la formula dell'output spline è `y_{i,j} = sum_{k=1}^{grid_size + spline_order} coef_{i,j,k} * B_{k}(x_i)`\n
        in cui, `i=1:in_size, j=1:out_size`. scritto in forma matriciale, `Y = B @ coef`,\n
        - `Y` con forma `(batch_size, in_size, out_size)`
        - `B` è il tensore delle basi B-spline `B_{k}(x_i)` con forma `(batch_size, in_size, grid_size + spline_order)`
        - `coef` è il tensore dei coefficienti spline con forma `(in_size, grid_size + spline_order, out_size)`

        `in_size` è una dimensione indipendente, `coef` trasforma il `grid_size + spline_order` in `out_size`

        Parametri:
        - `x: tf.Tensor` Tensore di input spline con forma `(batch_size, in_size)`
        - `y: tf.Tensor` Tensore di output spline con forma `(batch_size, in_size, out_size)`
        - `grid: tf.Tensor` Tensore della griglia spline con forma `(in_size, grid_size + 2 * spline_order + 1)`
        - `spline_order: int` Ordine spline
        - `l2_reg: float` Fattore di regolarizzazione L2 per il risolutore dei minimi quadrati
        - `fast: bool` flag che dice se utilizzare il risolutore veloce per il problema dei minimi quadrati

        Restituisce: `tf.Tensor` Tensore dei coefficienti spline con forma `(in_size, grid_size + spline_order, out_size)`
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