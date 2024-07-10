import tensorflow as tf
import keras

class DenseKAN(keras.Layer):
    """
    Costruisce un livello un livello di tipo KAN densamente connesso

    Parametri:
    - `units: int` Dimensione del tensore di output
    - `spline_order: int` Grado delle funzioni spline
    - `grid_size: int` Numero di intervalli della griglia
    - `grid_range: int` Estremi della griglia
    - `basis_activation: str` Funzione di attivazione
    - `use_bias: bool` Scelta se usare o no il bias
    - `kernel_initializer: str` Inizializzatore dei coefficienti delle spline
    - `scale_initializer: str` Inizializzatore dei fattori di scala delle spline
    - `bias_initializer: str` Inizializzatore del bias
    - `kernel_regularizer: str` Regolarizzatore dei coefficienti delle spline
    - `scale_regularizer: str` Regolarizzatore dei fattori di scala delle spline
    - `bias_regularizer: str` Regolarizzatore del bias
    - `activity_regularizer: str` Regolarizzatore dell'output
    - `kernel_constraint: str` Limitatore dei coefficienti delle spline
    - `scale_constraint: str` Limitatore dei fattori di scala delle spline
    - `bias_constraint: str` Limitatore del bias
    - `dtype: DType` Tipo dei coefficienti delle spline
    """

    def __init__(self,
        units: int,
        spline_order: int = 3,
        grid_size: int = 8,
        grid_range: tuple[float] = (-1, 1),
        basis_activation: str = 'silu',
        use_bias: bool = True,
        kernel_initializer: str | None = "random_normal",
        scale_initializer: str | None = "glorot_uniform",
        bias_initializer: str | None = "zeros",
        kernel_regularizer: str | None = None,
        scale_regularizer: str | None = None,
        bias_regularizer: str | None = None,
        activity_regularizer: str | None = None,
        kernel_constraint: str | None = None,
        scale_constraint: str | None = None,
        bias_constraint: str | None = None,
        dtype: tf.DType = tf.float32,
        **kwargs
    ):
        # Esegue il costruttore della superclasse
        super().__init__(dtype=dtype, activity_regularizer=activity_regularizer, **kwargs)

        # Salva i parametri nella classe e inizializza le variabili di classe
        self.units = units
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.basis_activation = keras.activations.get(basis_activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.scale_initializer = keras.initializers.get(scale_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.scale_regularizer = keras.regularizers.get(scale_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.scale_constraint = keras.constraints.get(scale_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        
        if self.units <= 0:
            raise ValueError("units must be positive")
        
        if self.spline_order < 0:
            raise ValueError("spline_order cannot be nagetive")
            
        self.spline_coefficient_size = self.grid_size - self.spline_order - 1
        if self.spline_coefficient_size <= self.spline_order:
            raise ValueError("grid_size must be at least 2*(spline_order + 1)")

    def build(self, input_shape):
        # Prende la dimensione di input e la salva nell'oggetto
        self.input_dim = input_shape[-1]

        # Definisce la matrice della griglia
        linspace = tf.linspace(self.grid_range[0], self.grid_range[1], self.grid_size)
        self.grid = tf.cast(tf.repeat(linspace[None, :], self.input_dim, axis=0), dtype=self.dtype)

        # Coefficienti di ogni spline-basis [c_i]
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.input_dim, self.spline_coefficient_size, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
        )

        # Coefficienti della spline complessiva [w_s]
        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(1, self.input_dim, self.units),
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer,
            constraint=self.scale_constraint,
            dtype=self.dtype,
        )

        # Coefficienti delle basis activation (bias) [w_b]
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(1, self.input_dim, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        self.built = True
    
    def call(self, inputs: tf.Tensor):
        inputs = tf.cast(inputs, dtype=self.dtype)
        spline_out = spline(inputs, self.grid, self.spline_kernel, self.spline_order, self.scale_factor, self.basis_activation, self.bias)
        
        # Per ogni neurone del livello, somma i risultati degli archi entranti        
        spline_out = tf.reduce_sum(spline_out, axis=-2)

        return spline_out
    
    def get_spline_list(self):
        if not self.built:
            raise Exception('Model not built')
        
        spline_list = []
        for i in range(self.input_dim):
            spline_list.append([])
            for j in range(self.units):
                func = Spline(self.grid[0], self.spline_kernel[i, :, j], self.spline_order, self.scale_factor[0, i, j], self.basis_activation, (self.bias[0, i, j] if self.use_bias else None))
                spline_list[i].append(func)
        return spline_list

    # Override metodo get_config, per aggiungere i nuovi parametri
    def get_config(self):
        config = super().get_config() # Recupera configurazione base del livello

        # Aggiunta parametri specifici di questo livello
        config.update({
            "units": self.units,
            "spline_order": self.spline_order,
            "grid_size": self.grid_size,
            "grid_range": self.grid_range,
            "basis_activation": self.basis_activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "scale_initializer": self.scale_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "scale_regularizer": self.scale_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "scale_constraint": self.scale_constraint,
            "bias_constraint": self.bias_constraint,
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Spline:
    def __init__(self, t: tf.Tensor, c: tf.Tensor, k: int, ws: int | tf.Tensor, b, wb: int | tf.Tensor | None) -> None:
        # Controlla validità parametri spline
        assert t.shape.rank == 1 and c.shape.rank == 1
        assert isinstance(k, int) and (isinstance(ws, int) or ws.shape.rank == 0) and (wb is None or isinstance(wb, int) or ws.shape.rank == 0)
        assert (len(c) >= len(t) - k - 1 >= k + 1)
        
        # Salva parametri nell'oggetto
        self.t = tf.expand_dims(t, axis=0)
        self.c = tf.reshape(c, (1, -1, 1))
        self.k = k
        self.ws = tf.reshape(ws, (1, 1, 1))
        self.b = b
        self.wb = tf.reshape(wb, (1, 1, 1)) if wb is not None else None
    
    def __call__(self, x: int | tf.Tensor):
        # Prepara x come un array
        if isinstance(x, tf.Tensor):
            orig_shape = x.shape
        new_x = tf.reshape(x, -1)

        # Calcola output spline 
        new_x = tf.expand_dims(new_x, axis=-1)
        out = spline(new_x, self.t, self.c, self.k, self.ws, self.b, self.wb)

        # Ridimensiona alla forma originale
        if isinstance(x, tf.Tensor):
            return tf.reshape(out, orig_shape)
        return int(out)


def spline(x: tf.Tensor, t: tf.Tensor, c: tf.Tensor, k: int, ws: tf.Tensor, b, wb: tf.Tensor) -> tf.Tensor:
    # Aggiunta di una dimensione sull'ultimo asse
    x = tf.expand_dims(x, axis=-1)

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

