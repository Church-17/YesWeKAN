import tensorflow as tf

def build_adaptive_grid(x: tf.Tensor,  grid_size: int,  spline_order: int,  grid_eps: float = 0.02,  margin: float = 0.01, dtype: tf.DType = tf.float32) -> tf.Tensor:
    """
        Costruisce la griglia adattiva basata sul tensore di input

        Parametri:
        - `x: tf.Tensor` Tensore di input con forma `(batch_size, in_size)`
        - `grid_size: int` Dimensione della griglia
        - `spline_order: int` Ordine dello spline
        - `grid_eps: float` Peso per combinare la griglia adattiva e la griglia uniforme
        - `margin: float` Margine per estendere la griglia
        - `dtype: DType` Tipo di dati per la griglia

        Restituisce: `tf.Tensor` Griglia adattiva con forma `(in_size, grid_size + 2 * spline_order + 1)`
    """

    # Formatta correttamente il vettore in input e lo ordina
    x = tf.cast(x, dtype=dtype) #converte x nel tipo dtype dato come input
    total = tf.shape(x)[0] #si salva la dimensione del batch
    n_records = tf.shape(x)[1]    #si salva il numero di elementi in input, cioè il numero di attributi 
    x_sorted = tf.sort(x, axis=0) 

    # Griglia adattiva - Basata sui dati | Dimensione = (grid_size+1, n_records)
    adaptive_idx = tf.cast(tf.linspace(0, total - 1, grid_size + 1), tf.int32) # Estremi degli intervalli
    grid_adaptive = tf.gather(x_sorted, adaptive_idx)                          # Mappa i dati adattandoli agli estremi degli intervalli

    # Griglia uniforme - Basata sull'intervallo di definizione | Dimensione = (grid_size+1, n_records)
    step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / grid_size                                # Appiezza degli intervalli (uniformi)         
    grid_uniform = x_sorted[0] - margin + tf.range(grid_size + 1, dtype=dtype)[:,None] * step   # Estremi degli intervalli
    grid_uniform = tf.reshape(grid_uniform, (6,n_records)) #fa il reshape della griglia uniforme    # MODIFICATO!

    # Combina la griglia adattiva con quella uniforme, pesandole con un coefficiente grid_eps | Dimensione = (grid_size+1, n_records)
    grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive

    # Estende la griglia a seconda del grado della spline, una spline di ordine d ha bisogno di una griglia di dimensione grid_size + 2*d
    # Dimensione = (grid_size + 2 * spline_order + 1, n_records)
    grid = tf.concat([
        grid[:1] - step * tf.range(spline_order, 0, -1, dtype=dtype)[:,None], # Aggiunge gli intervalli a sinistra 
        grid,
        grid[-1:] + step * tf.range(1, spline_order + 1, dtype=dtype)[:,None], # Aggiunge gli intervalli a destra
    ], axis=0)

    # Traspone la griglia per poterla utilizzare | Dimensione = (n_records, grid_size + 2 * spline_order + 1)
    grid = tf.transpose(grid)

    return grid #ritornala griglia

#print(build_adaptive_grid(tf.convert_to_tensor([list(range(100)) for _ in range(100)]), 5, 1))#questo serve stamparlo?