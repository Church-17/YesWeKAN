from typing import Callable
import tensorflow as tf
from keras import Layer
from scipy.interpolate import BSpline
import numpy as np
import keras


class DenseKAN(Layer):
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
    - `dtype: ` Tipo dei coefficienti delle spline
    """

    def __init__(self,
        units: int,
        spline_order: int = 3,
        grid_size: int = 5,
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
        self.spline_list = []

    def build(self, input_shape):
        # Prende la dimensione di input e la salva nell'oggetto
        self.input_dim = input_shape[-1]

        # Calcola parametri delle spline
        spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] -self.grid_range[0]

        # DA CAPIRE
        # Adatta la griglia al grado delle B-spline
        linspace_grid = tf.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size, # Estremo sinistro - grado_spline * ampiezza intervallo
            self.grid_range[1] + self.spline_order * bound / self.grid_size, # Estremo destro - grado_spline * ampiezza intervallo
            self.grid_size + 2 * self.spline_order + 1,                      # Numero totale di intervalli
        )

        # DA CAPIRE, MODIFICATO
        # Definisce un tensore con una griglia per ogni input
        self.grid = tf.cast(tf.repeat(linspace_grid[None, :], self.input_dim, axis=0), dtype=self.dtype)

        # Coefficienti di ogni spline-basis [Indicati con c_i nel paper]
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.input_dim, spline_basis_size, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        # Coefficienti della B-spline complessiva [Indicati con w_s nel paper]
        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(self.input_dim, self.units),
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer,
            constraint=self.scale_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        # Coefficienti delle Basis activation (bias) [Indicati con w_b nel paper]
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.built = True
        self._update_spline_list() 

    
    def call(self, inputs, *args, **kwargs):
        # Controlla gli input e ridimensiona gli input in un tensore 2D (-1, input_dim)
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.units]], axis=0)

        # Calcola l'output B-spline
        spline_out = self.calc_spline_output(inputs)

        # Calcola la base b(x) con forma (batch_size, input_dim)
        # Aggiunge la base a spline_out: phi(x) = c * (b(x) + spline(x)) utilizzando il broadcasting
        spline_out += tf.expand_dims(self.basis_activation(inputs), axis=-1)

        # Scala l'output
        spline_out *= tf.expand_dims(self.scale_factor, axis=0)
        
        # Aggrega l'output usando la somma (sulla dimensione input_dim) e ridimensiona alla forma originale        
        spline_out = tf.reshape(tf.reduce_sum(spline_out, axis=-2), output_shape)

        # Aggiunge il bias
        if self.use_bias:
            spline_out += self.bias

        return spline_out #ritorna la spline in output
    
    def _update_spline_list(self):
        self.spline_list = []
        for i in range(self.input_dim):
            for j in range(self.units):
                knots = self.grid[i]
                coeffs = self.spline_kernel[i, :, j]

                # Assicurarsi che il numero di coefficienti sia coerente con i nodi e il grado
                n = len(knots) - self.spline_order - 1
                if len(coeffs) > n:
                    coeffs = coeffs[:n]
                elif len(coeffs) < n:
                    coeffs = np.pad(coeffs, (0, n - len(coeffs)), mode='constant')

                try:
                    spline = BSpline(knots, coeffs, self.spline_order)
                    self.spline_list.append(spline)
                except ValueError as e:
                    print(f"Warning: Could not create spline for input {i}, unit {j}. Error: {str(e)}")
                    print(f"Knots shape: {knots.shape}, Coeffs shape: {coeffs.shape}, Degree: {self.spline_order}")


    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)  # shape dell input
        ndim = len(inputs.shape)  # Ottiene il numero di dimensioni del tensore
        try: #verifica se l'input sia bidimensionale e se non lo è genera un errore
            assert ndim >= 2
        except AssertionError:
            raise ValueError(f"expected min_ndim=2, found ndim={ndim}. Full shape received: {shape}")

        try:
            assert inputs.shape[-1] == self.input_dim # Controlla che l’ultima dimensione del tensore di input corrisponda a self.input_dim cioè la dimensione di input prevista
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self.input_dim}, found {shape[-1]}")

        # Reshape degli inputs in (-1, input_dim)
        orig_shape = shape[:-1]
        inputs = tf.reshape(inputs, (-1, self.input_dim))

        return inputs, orig_shape # Restituisce gli input ridimensionati e la forma originale
    
    def calc_spline_output(self, inputs: tf.Tensor):

        """
            Calcola la spline di output, ogni caratteristica di ogni campione viene mappata sulle caratteristiche di `out_size`,
            utilizzando `out_size` diverse funzioni di base B-spline, quindi la forma dell'output è `(batch_size, input_dim, out_size)`

            Parametri:
            - `inputs: tf.Tensor` Tensore con forma `(batch_size, input_dim)`
            
            Restituisce: `tf.Tensor` Tensore di output della spline con forma `(batch_size, input_dim, out_size)`
        """

        inputs = tf.cast(inputs, dtype=self.dtype)
        spline_in = calc_spline_values(inputs, self.grid, self.spline_order) # (B, input_dim, grid_basis_size)
        # Moltiplicazione matriciale con in coefficienti c_i: (batch, input_dim, grid_basis_size) @ (input_dim, grid_basis_size, out_size) -> (batch, input_dim, out_size)
        spline_out = tf.einsum("bik,iko->bio", spline_in, self.spline_kernel) #esegue una somma di einstein tra i due tensori e assegna il risultato a spline_out

        return spline_out
    
    # Override metodo get_config, per aggiungere i nuovi parametri
    def get_config(self):
        # Recupera configurazione base del livello
        config = super().get_config()

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
    

def calc_spline_values(x: tf.Tensor, grid: tf.Tensor, spline_order: int):
    """
    Calcola i valori B-spline per un tensore di input   

    Parameters
    - `x: tf.Tensor` tensore in input con forma `(batch_size, input_dim)`
    - `grid: tf.Tensor` tensore griglia di forma `(input_dim, grid_size + 2 * spline_order + 1)`
    - `spline_order: int` ordine delle spline

    Returns: `tf.Tensor` ritorna un tensore con una B-spline di forma (batch_size, input_dim, grid_size + spline_order)
    """

    # Il tensore in input deve essere di rango 2 (matrice 2D) | Dimensione = (batch_size, n_records), altrimenti si genera un errore
    assert x.shape.rank == 2
    
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

