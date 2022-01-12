from src.imports import *
import matplotlib.pyplot as plt


class Circuit(object):
    def __init__(self):
        pass

    def pure_state_tensor_to_density_matrix(
        self, pure_states: tf.Tensor, probs: tf.Tensor
    ) -> tf.Tensor:
        expanded_s = tf.expand_dims(pure_states, 1)  # (K, 1, 2**N)
        col_s = tf.transpose(expanded_s, [0, 2, 1])  # (K, 2**N, 1)
        adj_s = tf.math.conj(expanded_s)  # (K, 1, 2**N)
        prod = tf.linalg.matmul(col_s, adj_s)  # (K, 2**N, 2**N) ,  A * A^dagger
        min_ = np.min(tf.linalg.trace(prod).numpy())

        density_matrix = tf.reduce_sum(
            prod * tf.cast(tf.expand_dims(tf.expand_dims(probs, 1), 2), tf.complex64), 0
        )
        return density_matrix

    def create_circuits(self) -> None:
        Add_Circuit = tfq.layers.AddCircuit()
        self.unitary = self.unitary_layer(layer=1)
        self.depolarization = self.depolarizing_layer(layer=1)
        self.circuit = Add_Circuit(self.unitary, append=self.depolarization)

        for l in range(2, 1 + self.L):
            unitary = self.unitary_layer(layer=l)
            self.unitary = Add_Circuit(self.unitary, append=unitary)
            self.circuit = Add_Circuit(self.circuit, append=unitary)

            depolarization = self.depolarizing_layer(layer=l)
            self.depolarization = Add_Circuit(
                self.depolarization, append=depolarization
            )
            self.circuit = Add_Circuit(self.circuit, append=depolarization)

        self.circuits = tf.tile(self.circuit, [self.K])
        self.gamma_symbols_str = self.gamma_symbols
        self.gamma_symbols = tf.constant([str(s) for s in self.gamma_symbols])
        self.error_symbols = tf.constant([str(s) for s in self.error_symbols])
        self.symbols = tf.concat([self.gamma_symbols, self.error_symbols], 0)

        self.unitary = tfq.from_tensor(self.unitary)[0]
        self.depolarization = tfq.from_tensor(self.depolarization)[0]
        self.circuit = tfq.from_tensor(self.circuit)[0]

    def unitary_layer(self, layer: int) -> cirq.Circuit:
        unitary = cirq.Circuit()

        if self.ansatz == "qaoa-r":
            if layer == 1:
                unitary = cirq.Circuit([cirq.H(qubit) for qubit in self.qubits])
            gamma = sympy.Symbol("Rzz_L_{0}".format(layer))
            self.gamma_symbols.append(gamma)
            for n, q in enumerate(self.qubits):
                if n == self.N - 1:
                    unitary += cirq.ZZPowGate(exponent=gamma).on(q, self.qubits[0])
                else:
                    unitary += cirq.ZZPowGate(exponent=gamma).on(q, self.qubits[n + 1])

            gamma = sympy.Symbol("Rz_L_{0}".format(layer))
            self.gamma_symbols.append(gamma)
            unitary += cirq.Circuit(
                (cirq.rz(gamma)(self.qubits[n]) for n in range(self.N))
            )
            gamma = sympy.Symbol("Rx_L_{0}".format(layer))
            self.gamma_symbols.append(gamma)
            unitary += cirq.Circuit(
                (cirq.rx(gamma)(self.qubits[n]) for n in range(self.N))
            )

        elif self.ansatz == "qaoa-f":
            if layer == 1:
                unitary = cirq.Circuit([cirq.H(qubit) for qubit in self.qubits])
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rzz_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                if n == self.N - 1:
                    unitary += cirq.ZZPowGate(exponent=gamma).on(q, self.qubits[0])
                else:
                    unitary += cirq.ZZPowGate(exponent=gamma).on(q, self.qubits[n + 1])

            moment = []
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rz_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                moment.append(cirq.rz(gamma)(q))
            unitary += cirq.Circuit(cirq.Moment(moment))

            moment = []
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rx_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                moment.append(cirq.rx(gamma)(q))
            unitary += cirq.Circuit(cirq.Moment(moment))

        elif self.ansatz == "qaoa-all":
            if layer == 1:
                unitary = cirq.Circuit([cirq.H(qubit) for qubit in self.qubits])
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rzz_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                if n == self.N - 1:
                    unitary += cirq.ZZPowGate(exponent=gamma).on(q, self.qubits[0])
                else:
                    unitary += cirq.ZZPowGate(exponent=gamma).on(q, self.qubits[n + 1])
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rxx_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                if n == self.N - 1:
                    unitary += cirq.XXPowGate(exponent=gamma).on(q, self.qubits[0])
                else:
                    unitary += cirq.XXPowGate(exponent=gamma).on(q, self.qubits[n + 1])
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Ryy_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                if n == self.N - 1:
                    unitary += cirq.YYPowGate(exponent=gamma).on(q, self.qubits[0])
                else:
                    unitary += cirq.YYPowGate(exponent=gamma).on(q, self.qubits[n + 1])

            moment = []
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rx_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                moment.append(cirq.rx(gamma)(q))
            unitary += cirq.Circuit(cirq.Moment(moment))

        elif self.ansatz == "rot-cnot":
            moment = []
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rz1_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                moment.append(cirq.rz(gamma)(q))
            unitary += cirq.Circuit(cirq.Moment(moment))

            moment = []
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Ry_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                moment.append(cirq.ry(gamma)(q))
            unitary += cirq.Circuit(cirq.Moment(moment))

            moment = []
            for n, q in enumerate(self.qubits):
                gamma = sympy.Symbol("Rz1_L_{0}_q_{1}".format(layer, n + 1))
                self.gamma_symbols.append(gamma)
                moment.append(cirq.rz(gamma)(q))
            unitary += cirq.Circuit(cirq.Moment(moment))

            for n, q in enumerate(self.qubits):
                if n == self.N - 1:
                    unitary += cirq.CNOT(q, self.qubits[0])
                else:
                    unitary += cirq.CNOT(q, self.qubits[n + 1])

        unitary_t = tfq.convert_to_tensor([unitary])
        return unitary_t

    def depolarizing_layer(self, layer: int) -> cirq.Circuit:
        depolarization = cirq.Circuit()
        for gate, gate_str in [(cirq.X, "X"), (cirq.Y, "Y"), (cirq.Z, "Z")]:
            for n, q in enumerate(self.qubits):
                symbol = sympy.Symbol("D_{0}_q{1}_L{2}".format(gate_str, n, layer))
                self.error_symbols.append(symbol)
                depolarization += gate(q) ** symbol

        depolarization_t = tfq.convert_to_tensor([depolarization])
        return depolarization_t

    def gamma_init(self, loc: float = 0) -> tf.Variable:
        p_gammas = tfp.distributions.Uniform(
            low=0.0001, high=0.05
        )  # Kastoryano et al. (2020)
        gammas = p_gammas.sample(sample_shape=(1, len(self.gamma_symbols)))
        return tf.Variable(gammas)

    def sample_errors(self, p_err: float = None, K: int = None) -> tf.Tensor:
        p_err = self.p_err if p_err is None else p_err
        K = self.K if K is None else K
        p_errs = (
            np.tile(self.p_errs, (self.L * 3))
            if self.multilambda
            else np.array([(p_err / 3) for _ in range(self.L * self.N * 3)])
        )  # multiply by 3 because X,Y,Z gate
        bern_dist = tfp.distributions.Bernoulli(probs=p_errs, dtype=tf.float32)
        errors = tf.squeeze(bern_dist.sample(K))
        return errors

    def sample_errors_i(self, i, p_i: float) -> tf.Tensor:
        K = self.K
        p_dist = self.p_errs
        p_dist[i] = p_i
        p_errs = np.tile(p_dist, (self.L * 3))  # multiply by 3 because X,Y,Z gate
        bern_dist = tfp.distributions.Bernoulli(probs=p_errs, dtype=tf.float32)
        errors = bern_dist.sample(K)
        return errors

    def plot_history(self, save: bool = False):
        f = plt.figure(figsize=(18, 10))
        for i_c, col in enumerate(["F", "G", "H", "S", "p_err"]):
            plt.subplot(2, 3, i_c + 1)
            if col == "G" or col == "S":
                plt.plot(self.history[col], "--*", label=col)
                if not self.multilambda:
                    plt.plot(
                        self.history[col + "_approx"],
                        "--*",
                        label=r"$\hat{" + col + "}$",
                    )
                    min_ = np.minimum(
                        np.min(self.history[col]), np.min(self.history[col + "_approx"])
                    )
                    max_ = np.maximum(
                        np.max(self.history[col]), np.max(self.history[col + "_approx"])
                    )
                if col == "G":
                    plt.plot(
                        [self.G_target] * len(self.history[col].values),
                        "black",
                        linewidth=2,
                        label="Target",
                    )
                if col == "S":
                    plt.plot(
                        [self.S_target] * len(self.history[col].values),
                        "black",
                        linewidth=2,
                        label="Target",
                    )
                plt.legend()
            elif col == "F":
                plt.plot(
                    self.history[col],
                    "--*",
                    label=col + " ({:.2E})".format(np.max(self.history[col])),
                )
                plt.plot(
                    self.history["T"],
                    "--*",
                    label="T" + " ({:.2E})".format(np.min(self.history["T"])),
                )
                plt.plot(
                    self.history["RMSE"],
                    "--*",
                    label="RMSE" + " ({:.2E})".format(np.min(self.history["RMSE"])),
                )
                min_ = 0
                max_ = 1
                plt.ylim([min_, max_])
                plt.legend()
            else:
                plt.plot(self.history[col], "--*")
                plt.ylabel(col)
                if col == "H":
                    plt.plot(
                        [self.H_target] * len(self.history[col].values),
                        "black",
                        linewidth=2,
                        label="Target",
                    )
                    plt.plot(
                        [self.H_0] * len(self.history[col].values),
                        linestyle="--",
                        color="grey",
                        linewidth=1,
                        label=r"$E_0$",
                    )
                    plt.plot(
                        [self.H_1] * len(self.history[col].values),
                        linestyle="--",
                        color="lightgrey",
                        linewidth=1,
                        label=r"$E_1$",
                    )
                    plt.legend()
                min_ = np.min(self.history[col])
                max_ = np.max(self.history[col])
            plt.xlabel("Epochs")
            if col == "p_err":
                plt.ylabel(r"||$\lambda||_2$")
                plt.ticklabel_format(axis="y", scilimits=(0, 1))
                # plt.ylim([0,1])
            else:
                plt.ticklabel_format(axis="y", style="sci", scilimits=(min_, max_))

        plt.subplot(2, 3, 6)
        plt.plot(
            np.abs(self.history["grad_H_p_err"]), "--*", label=r"$|\nabla_{\lambda} H|$"
        )
        plt.plot(
            np.abs(self.history["grad_TS_p_err"]),
            "--*",
            label=r"$|\beta^{-1}\nabla_{\lambda} S|$",
        )
        plt.plot(
            self.history["ngrad_H_gamma"], "--*", label=r"$||\nabla_{\gamma} H||_2$"
        )
        plt.xlabel("Epochs")
        plt.legend()
        plt.ticklabel_format(axis="y", style="sci")
        plt.yscale("log")

        if save:
            f.savefig(self.savepth + "history---" + self.settings + ".pdf")
        plt.close()

    def plot_density_matrices(self, save: bool = False):

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        rho = np.absolute(self.rho)  # np.log(+ 1e-20)
        target_rho = np.absolute(self.target_rho)  # np.log(+ 1e-20)
        min_ = np.minimum(np.min(rho), np.min(target_rho))
        max_ = np.maximum(np.max(rho), np.max(target_rho))
        im = axes.flat[0].imshow(rho, vmin=min_, vmax=max_, aspect="auto")
        axes.flat[0].grid(False)
        axes.flat[0].set_xticks([])
        axes.flat[0].set_yticks([])
        axes.flat[0].set_title("Circuit density")

        im = axes.flat[1].imshow(target_rho, vmin=min_, vmax=max_, aspect="auto")
        axes.flat[1].grid(False)
        axes.flat[1].set_xticks([])
        axes.flat[1].set_yticks([])
        axes.flat[1].set_title("Target density")

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$|\rho_{ij}|$", fontsize=20)
        if save:
            fig.savefig(self.savepth + "density-matrices---" + self.settings + ".pdf")

        plt.show()

    def run_circuit(self, gammas_: np.ndarray = None):
        gammas = tf.identity(self.gammas) if gammas_ is None else gammas_
        if self.DMS:
            simulator = DensityMatrixSimulator(noise=cirq.depolarize(self.p_err))
            rho = simulator.simulate(
                self.unitary,
                cirq.ParamResolver(
                    {
                        symbol: self.gammas.numpy()[0][i]
                        for i, symbol in enumerate(self.gamma_symbols_str)
                    }
                ),
            ).final_density_matrix
            return tf.convert_to_tensor(rho)
        else:
            values = tf.concat(
                [tf.tile(gammas, tf.constant([self.K, 1])), self.errors], 1
            )
            noisy_states = tfq.layers.State()(
                self.circuits, symbol_names=self.symbols, symbol_values=values
            ).to_tensor()
            probs = tf.ones([self.K], dtype=tf.float32) / float(self.K)
            noisy_dm = self.pure_state_tensor_to_density_matrix(noisy_states, probs)
            return noisy_dm
