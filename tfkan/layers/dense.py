from typing import Callable
import tensorflow as tf
from keras import Layer

from ..ops.spline import fit_spline_coef, calc_spline_values
from ..ops.grid import build_adaptive_grid


class DenseKAN(Layer):
    def __init__(self,
        units: int,
        use_bias: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float] | list[float] = (-1.0, 1.0),
        spline_initialize_stddev: float = 0.1,
        basis_activation: str | Callable = 'silu',
        dtype = tf.float32,
        **kwargs
    ):
        # Esegue il costruttore della superclasse
        super().__init__(dtype=dtype, **kwargs)

        # Save parameters in class
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.basis_activation = basis_activation
        self.use_bias = use_bias

        # Initialize parameters
        self.spline_initialize_stddev = spline_initialize_stddev

    def build(self, input_shape):
        # Controllo se l'input è un vettore o un tensore n-D
        if isinstance(input_shape, int):
            in_size = input_shape
        else:
            in_size = input_shape[-1]

        self.in_size = in_size
        self.spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] -self.grid_range[0]

        # Adatta la griglia al grado della B-spline
        self.grid = tf.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size, # Estremo sinistro - grado_spline * ampiezza intervallo
            self.grid_range[1] + self.spline_order * bound / self.grid_size, # Estremo destro - grado_spline * ampiezza intervallo
            self.grid_size + 2 * self.spline_order + 1                       # Numero totale di intervalli
        )
        # Definisce un tensore con una griglia per ogni input
        self.grid = tf.repeat(self.grid[None, :], in_size, axis=0)
        self.grid = tf.Variable(
            initial_value=tf.cast(self.grid, dtype=self.dtype),
            trainable=False,
            dtype=self.dtype,
            name="spline_grid"
        )

        # Coefficienti di ogni spline-basis [Indicati con c_i nel paper]
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.spline_initialize_stddev),
            trainable=True,
            dtype=self.dtype
        )

        # Coefficienti della B-spline complessiva [Indicati con w_s nel paper]
        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(self.in_size, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype
        )

        # Basis activation [Indicata con b(x) nel paper]
        if isinstance(self.basis_activation, str):
            self.basis_activation = tf.keras.activations.get(self.basis_activation)
        elif not callable(self.basis_activation):
            raise ValueError(f"expected basis_activation to be str or callable, found {type(self.basis_activation)}")

        # Coefficienti delle Basis activation (bias) [Indicati con w_b nel paper]
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=self.dtype
            )
        else:
            self.bias = None

        self.built = True

    
    def call(self, inputs, *args, **kwargs):
        # check the inputs, and reshape inputs into 2D tensor (-1, in_size)
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.units]], axis=0)

        # calculate the B-spline output
        spline_out = self.calc_spline_output(inputs)

        # calculate the basis b(x) with shape (batch_size, in_size)
        # add basis to the spline_out: phi(x) = c * (b(x) + spline(x)) using broadcasting
        spline_out += tf.expand_dims(self.basis_activation(inputs), axis=-1)

        # scale the output
        spline_out *= tf.expand_dims(self.scale_factor, axis=0)
        
        # aggregate the output using sum (on in_size dim) and reshape into the original shape
        spline_out = tf.reshape(tf.reduce_sum(spline_out, axis=-2), output_shape)

        # add bias
        if self.use_bias:
            spline_out += self.bias

        return spline_out
    
    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        ndim = len(inputs.shape)  # Ottiene il numero di dimensioni del tensore
        try:
            assert ndim >= 2
        except AssertionError:
            raise ValueError(f"expected min_ndim=2, found ndim={ndim}. Full shape received: {shape}")

        try:
            assert inputs.shape[-1] == self.in_size # Controlla la dimensione di input
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self.in_size}, found {shape[-1]}")

        # Reshape degli inputs in (-1, in_size)
        orig_shape = shape[:-1]
        inputs = tf.reshape(inputs, (-1, self.in_size))

        return inputs, orig_shape
    
    def calc_spline_output(self, inputs: tf.Tensor):
        """
        Calculate the spline output, each feature of each sample is mapped to `out_size` features,
        using `out_size` different B-spline basis functions, so the output shape is `(batch_size, in_size, out_size)`

        Parameters:
        - `inputs: tf.Tensor` Tensor with shape `(batch_size, in_size)`
        
        Returns: `tf.Tensor` Spline output tensor with shape `(batch_size, in_size, out_size)`
        """
        inputs = tf.cast(inputs, dtype=tf.float64)
        # Calcola le basi B-spline
        spline_in = calc_spline_values(inputs, self.grid, self.spline_order) # (B, in_size, grid_basis_size)
        # Moltiplicazione matriciale con in coefficienti c_i: (batch, in_size, grid_basis_size) @ (in_size, grid_basis_size, out_size) -> (batch, in_size, out_size)
        spline_out = tf.einsum("bik,iko->bio", spline_in, self.spline_kernel)

        return spline_out

    def update_grid_from_samples(self, inputs: tf.Tensor, margin: float = 0.01, grid_eps: float = 0.01):
        inputs = tf.cast(inputs, dtype=tf.float64)
        # Controlla gli input e fa il reshape in un vettore 2D | Dimensione = (-1, in_size)
        inputs, _ = self._check_and_reshape_inputs(inputs)

        # Calcola l'approssimazione delle spline
        spline_out = self.calc_spline_output(inputs)

        # Ricalcola la griglia
        grid = build_adaptive_grid(inputs, self.grid_size, self.spline_order, grid_eps, margin, self.dtype)
        
        # Ricalcola i coefficienti c_i per la nuova dimensione della griglia
        updated_kernel = fit_spline_coef(inputs, spline_out, grid, self.spline_order)

        # Ridefinisce la griglia e i coefficienti
        self.grid.assign(grid)
        self.spline_kernel.assign(updated_kernel)
    

    def extend_grid_from_samples(self, inputs: tf.Tensor, extend_grid_size: int, margin: float = 0.01, grid_eps: float = 0.01, l2_reg: float = 0, fast: bool = True):
        
        inputs = tf.cast(inputs, dtype=tf.float64)
        # Verifica che la griglia estesa sia di dimensione maggiore rispetto alla precedente
        try:
            assert extend_grid_size >= self.grid_size
        except AssertionError:
            raise ValueError(f"expected extend_grid_size > grid_size, found {extend_grid_size} <= {self.grid_size}")

        # Controlla gli input e fa il reshape in un vettore 2D | Dimensione = (-1, in_size)
        inputs, _ = self._check_and_reshape_inputs(inputs)

        # Calcola l'approssimazione delle spline
        spline_out = self.calc_spline_output(inputs)

        # Ricalcola la griglia | Dimensione = (n_inputs, extend_grid_size + 2 * spline_order + 1)
        grid = build_adaptive_grid(inputs, extend_grid_size, self.spline_order, grid_eps, margin, self.dtype)

        # Aggiorna i coefficienti c_i delle B-spline
        updated_kernel = fit_spline_coef(inputs, spline_out, grid, self.spline_order, l2_reg, fast)

        # Ridefinizione della griglia
        delattr(self, "grid")
        self.grid = tf.Variable(
            initial_value=tf.cast(grid, dtype=self.dtype),
            trainable=False,
            dtype=self.dtype,
            name="spline_grid"
        )

        # Ridefinizione dei parametri c_i
        self.grid_size = extend_grid_size
        self.spline_basis_size = extend_grid_size + self.spline_order
        delattr(self, "spline_kernel")
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.Constant(updated_kernel),
            trainable=True,
            dtype=self.dtype
        )

    # Aggiornamento della configurazione
    def get_config(self):
        config = super(DenseKAN, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "spline_initialize_stddev": self.spline_initialize_stddev,
            "basis_activation": self.basis_activation
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
