import tensorflow as tf
import numpy as np

class Spline:
    """
    Questa classe costruisce una funzione di attivazione spline usata da un livello DenseKAN

    Parametri:
    - `t: Tensor` Array di nodi, ovvero di punti in cui le singole funzioni B-Spline di uniscono
    - `c: Tensor` Array dei coefficienti
    - `k: int` Grado della spline
    - `ws: int` Fattori di scala
    - `b:` basis function
    - `wb:` Coefficienti del bias
    """

    def __init__(self, t: tf.Tensor, c: tf.Tensor, k: int, ws: tf.Tensor, b, wb: tf.Tensor | None, dtype: tf.DType = tf.float32) -> None:
        # Controlla validità parametri spline
        t = tf.convert_to_tensor(t, dtype)
        c = tf.convert_to_tensor(c, dtype)
        ws = tf.convert_to_tensor(ws, dtype)
        wb = tf.convert_to_tensor(wb, dtype) if wb is not None else None
        assert t.shape.rank == 1 and c.shape.rank == 1
        assert isinstance(k, int) and k >= 0 and ws.shape.rank == 0 and (wb is None or ws.shape.rank == 0)
        check_sort = t.numpy()
        assert np.all(check_sort[:-1] <= check_sort[1:])
        assert (len(c) >= len(t) - k - 1 >= k + 1)
        
        # Salva parametri nell'oggetto, convertendoli alla shape usata dalla funzione spline
        self.t = tf.expand_dims(t, axis=0)
        self.c = tf.reshape(c, (1, -1, 1))
        self.k = k
        self.ws = tf.reshape(ws, (1, 1, 1))
        self.b = tf.keras.activations.get(b)
        self.wb = tf.reshape(wb, (1, 1, 1)) if wb is not None else None
        self.dtype = dtype
    
    def __call__(self, x: float | tf.Tensor) -> float | tf.Tensor:
        # Prepara x come un array
        x = tf.convert_to_tensor(x, self.dtype)
        orig_shape = x.shape
        x = tf.reshape(x, -1)

        # Calcola output spline 
        x = tf.expand_dims(x, axis=-1)
        out = spline(x, self.t, self.c, self.k, self.ws, self.b, self.wb)

        # Ridimensiona alla forma originale
        out = tf.reshape(out, orig_shape)
        if out.shape == ():
            return float(out)
        return out

# Funzione che calcola effettivamente il valore della spline dati tutti i parametri e gli input
def spline(x: tf.Tensor, t: tf.Tensor, c: tf.Tensor, k: int, ws: tf.Tensor, b, wb: tf.Tensor | None) -> tf.Tensor:
    # Aggiunta di una dimensione sull'ultimo asse
    x = tf.expand_dims(x, -1)

    # Definizione della B-spline di grado 0
    spline_out = tf.logical_and(tf.greater_equal(x, t[:, :-1]), tf.less(x, t[:, 1:]))
    spline_out = tf.cast(spline_out, x.dtype)
    
    # Definizione ricorsiva delle B-spline fino al grado voluto
    for k in range(1, k+1):
        spline_out = (
            (x - t[:, :-(k+1)]) / (t[:, k:-1] - t[:, :-(k+1)]) * spline_out[:, :, :-1]
        ) + (
            (t[:, k+1:] - x) / (t[:, k+1:] - t[:, 1:-k]) * spline_out[:, :, 1:]
        )

    # Combinazione lineare delle B-spline con i coefficienti
    spline_out = tf.einsum("bik,iko->bio", spline_out, c)

    # Moltiplica ogni output delle spline per il suo scale_factor
    spline_out *= ws

    # Calcola b(x)
    basis = tf.repeat(b(x), spline_out.shape[-1], axis=-1)

    # Se viene dato wb, calcola wb * b(x)
    if wb is not None:
        basis *= wb

    # Somma ws * spline(x) con wb * b(x)
    spline_out += basis

    return spline_out

