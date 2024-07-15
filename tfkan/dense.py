import tensorflow as tf

from .spline import Spline, spline

class DenseKAN(tf.keras.Layer):
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
        kernel_initializer: tf.keras.Initializer | str | None = tf.keras.initializers.RandomNormal(stddev=0.1),
        scale_initializer: tf.keras.Initializer | str | None = tf.keras.initializers.Ones(),
        bias_initializer: tf.keras.Initializer | str | None = tf.keras.initializers.GlorotNormal(),
        kernel_regularizer: tf.keras.Regularizer | str | None = None,
        scale_regularizer: tf.keras.Regularizer | str | None = None,
        bias_regularizer: tf.keras.Regularizer | str | None = None,
        activity_regularizer: tf.keras.Regularizer | str | None = None,
        kernel_constraint: tf.keras.constraints.Constraint | str | None = None,
        scale_constraint: tf.keras.constraints.Constraint | str | None = None,
        bias_constraint: tf.keras.constraints.Constraint | str | None = None,
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
        self.basis_activation = tf.keras.activations.get(basis_activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.scale_initializer = tf.keras.initializers.get(scale_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.scale_regularizer = tf.keras.regularizers.get(scale_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.scale_constraint = tf.keras.constraints.get(scale_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        
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
                func = Spline(self.grid[0], self.spline_kernel[i, :, j], self.spline_order, self.scale_factor[0, i, j], self.basis_activation, (self.bias[0, i, j] if self.use_bias else None), self.dtype)
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

