"""Controle da garra RG2: abrir/fechar (via sinal de cena) + 'fake grasp'
(anexa o objeto detectado pelo sensor de proximidade da garra, tecnica
sugerida no proprio script do modelo RG2)."""

_SETTLE_STEPS = 50  # ~2.5s a 20Hz -- a garra leva ~2s para abrir/fechar totalmente


class RG2Gripper:
    def __init__(self, conn):
        self.conn = conn
        self.sim = conn.sim
        self.attach_point = self.sim.getObject("/UR5/RG2/attachPoint")
        self.prox_sensor = self.sim.getObject("/UR5/RG2/attachProxSensor")
        self._attached_shape = None

    def _set_open_signal(self, open_: bool):
        # o script Lua da RG2 le isso via sim.getIntProperty(handle_scene,
        # 'signal.RG2_open', ...) -- sim.setInt32Signal (API de sinais antiga)
        # NAO escreve nesse mesmo lugar nesta versao do CoppeliaSim, entao o
        # sinal nunca chegava ao script (a garra ficava sempre fechando, que
        # e o comportamento padrao do script quando o sinal esta ausente).
        self.sim.setIntProperty(self.sim.handle_scene, "signal.RG2_open", 1 if open_ else 0)

    def open(self):
        self._set_open_signal(True)
        for _ in range(_SETTLE_STEPS):
            self.conn.step()
        if self._attached_shape is not None:
            self.sim.setObjectParent(self._attached_shape, -1, True)
            self._attached_shape = None

    def close(self):
        self._set_open_signal(False)
        for _ in range(_SETTLE_STEPS):
            self.conn.step()
        self._try_attach()

    def _try_attach(self):
        result, dist_data, det_point, obj_handle, *_ = self.sim.checkProximitySensor(
            self.prox_sensor, self.sim.handle_all
        )
        if result == 1 and obj_handle != -1:
            is_dynamic = not self.sim.getObjectInt32Param(
                obj_handle, self.sim.shapeintparam_static
            )
            is_respondable = self.sim.getObjectInt32Param(
                obj_handle, self.sim.shapeintparam_respondable
            )
            if is_dynamic and is_respondable:
                self.sim.setObjectParent(obj_handle, self.attach_point, True)
                self._attached_shape = obj_handle

    @property
    def is_holding(self):
        return self._attached_shape is not None
