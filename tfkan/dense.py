from scipy.interpolate import BSpline
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
        dtype: tf.DType = tf.float64,
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
    
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=self.dtype)

        # Calcola l'output delle spline
        spline_out = spline(inputs, self.grid, self.spline_kernel, self.spline_order)

        # Moltiplica ogni output delle spline per il suo scale_factor w_s
        spline_out *= self.scale_factor

        # Calcola la base b(x) con forma (batch_size, input_dim) e trasformala in matrice (batch_size, input_dim, output_dim)
        basis = tf.repeat(tf.expand_dims(self.basis_activation(inputs), axis=-1), self.units, axis=-1)

        # Se c'è il bias, calcola w_b * b(x)
        if self.use_bias:
            basis *= self.bias
            
        # Somma w_s * spline(x) con w_b * b(x)
        spline_out += basis
        
        # Per ogni neurone del livello, somma i risultati degli archi entranti        
        spline_out = tf.reduce_sum(spline_out, axis=-2)

        return spline_out
    
    def get_spline_list(self):
        if not self.built:
            raise Exception('Model not built')
        
        spline_list = []
        for i in range(self.input_dim):
            for j in range(self.units):
                knots = self.grid[i]
                coeffs = self.spline_kernel[i, :, j]
                func = Spline(self.grid[0], self.spline_kernel[i, :, j], self.spline_order)
                # func = BSpline(knots, coeffs, self.spline_order)
                spline_list.append(func)
        return spline_list

    # Override metodo get_config, per aggiungere i nuovi parametri
    def get_config(self):
        config = super().get_config() # Recupera configurazione base del livello

        # Aggiunta parametri specifici di questo livello
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "basis_activation": self.basis_activation
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Spline:
    def __init__(self, knots: tf.Tensor, coeff: tf.Tensor, degree: int) -> None:
        self.knots = tf.expand_dims(knots, axis=0)
        self.coeff = tf.expand_dims(tf.expand_dims(coeff, axis=0), axis=-1)
        self.degree = degree
    
    def __call__(self, x: tf.Tensor):
        print(x.shape, self.knots.shape, self.coeff.shape)
        x = tf.expand_dims(x, axis=-1)
        out = spline(x, self.knots, self.coeff, self.degree)
        return tf.reshape(out, [-1])

def spline(x: tf.Tensor, grid: tf.Tensor, coeff: tf.Tensor, degree: int) -> tf.Tensor:
    # Aggiunta di una dimensione sull'ultimo asse
    x = tf.expand_dims(x, axis=-1)

    # Definizione della B-spline di grado 0
    bases = tf.logical_and(tf.greater_equal(x, grid[:, :-1]), tf.less(x, grid[:, 1:]))
    bases = tf.cast(bases, x.dtype)
    
    # Definizione ricorsiva delle B-spline fino al grado voluto
    for k in range(1, degree+1):
        bases = (
            (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
        ) + (
            (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:]
        )

    return tf.einsum("bik,iko->bio", bases, coeff)

