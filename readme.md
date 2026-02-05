# Security Analysis on Device-Independent Quantum Key Distribution Protocols with Routed Bell Tests
### Abstract
In this thesis, we simulate a routed Bell-based Device-Independent Quantum Key Distribu-
tion (DIQKD) protocol under various noise conditions and analyze its effects on the keyrate.
The protocol leverages entangled qubit pairs and Bell tests to establish secure keys between
two parties, Alice and Bob, without relying on the trustworthiness of their devices. We
implement the protocol using a Python library called qiskit, simulating various scenarios
with different levels of channel loss and noise. Our simulations reveal that the routed
Bell-based DIQKD protocol can achieve positive keyrates under noise, although much more
modest than in ideal conditions. We further run the protocol on IBM Quantum hardware
to validate our simulation results and assess the practical feasibility of the protocol. We
find that the keyrates are significantly affected by realistic noise and loss, highlighting
the challenges in implementing DIQKD protocols in practice, as no positive keyrate could
be achieved under the tested configurations. Therefore, we conclude that the impact of
different noise models, including depolarizing noise and detector inefficiencies, on the
keyrate are quite substantial. Moreover, we identify necessary adjustments to the SDP
(Semi-Definite Programming) formulations used in the security analysis to accommodate
finite-size effects and realistic noise models. We also explore the possibility of cross-talk
effects impacting the keyrate, finding more research is needed to fully understand these
effects. Finally, we discuss potential improvements and future research directions for
enhancing the robustness and efficiency of routed Bell-based DIQKD protocols.