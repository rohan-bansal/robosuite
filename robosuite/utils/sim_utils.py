"""
Collection of useful simulation utilities
"""

import mujoco
import numpy as np

from robosuite.models.base import MujocoModel


def _to_geom_name_list(geoms):
    """
    Normalizes @geoms (a geom name, list of geom names, MujocoModel, or None) into a list of geom
    name strings (or None). Mirrors the input handling used by :func:`check_contact`.
    """
    if geoms is None:
        return None
    if type(geoms) is str:
        return [geoms]
    if isinstance(geoms, MujocoModel):
        return list(geoms.contact_geoms)
    return list(geoms)


def check_contact(sim, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.
    Args:
        sim (MjSim): Current simulation object
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment
    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = sim.model.geom_id2name(contact.geom1) in geoms_1
        c2_in_g2 = sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = sim.model.geom_id2name(contact.geom2) in geoms_1
        c1_in_g2 = sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


def get_contacts(sim, model):
    """
    Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
    geom names currently in contact with that model (excluding the geoms that are part of the model itself).
    Args:
        sim (MjSim): Current simulation model
        model (MujocoModel): Model to check contacts for.
    Returns:
        set: Unique geoms that are actively in contact with this model.
    Raises:
        AssertionError: [Invalid input type]
    """
    # Make sure model is MujocoModel type
    assert isinstance(model, MujocoModel), "Inputted model must be of type MujocoModel; got type {} instead!".format(
        type(model)
    )
    contact_set = set()
    for contact in sim.data.contact[: sim.data.ncon]:
        # check contact geom in geoms; add to contact set if match is found
        g1, g2 = sim.model.geom_id2name(contact.geom1), sim.model.geom_id2name(contact.geom2)
        if g1 in model.contact_geoms and g2 not in model.contact_geoms:
            contact_set.add(g2)
        elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
            contact_set.add(g1)
    return contact_set


def get_contact_forces(sim, geoms_1=None, geoms_2=None, actor_geoms=None, normalize_normal=True):
    """
    Per-contact analog to ManiSkill's contact API. Iterates over MuJoCo's active contact array and,
    for each contact, returns its world-frame position, contact normal, and full force vector (computed
    via ``mujoco.mj_contactForce``). Optionally filters to contacts between two geom groups -- e.g. the
    gripper fingerpads vs. an object (grasp) or a fingertip vs. a target (push).

    IMPORTANT: Contact forces are only populated after the constraint solver runs. Call this *after*
    ``env.step()`` / ``sim.step()`` (or a ``sim.forward()`` on a state that was stepped into), not before.

    Sign convention (verified empirically against a box resting on a table): by default the returned
    ``force`` is the force acting on ``geom2``, i.e. the force ``geom1`` exerts on ``geom2``, pushing
    ``geom2`` away from ``geom1`` (the contact ``normal`` points from ``geom1`` toward ``geom2``).
    MuJoCo orders ``geom1``/``geom2`` by internal geom id, which is arbitrary relative to "robot vs.
    object". To get an intuitive direction, pass @actor_geoms (see below) so the force is always
    expressed as the force the actor exerts on whatever it touches -- e.g. the force the gripper pushes
    an object with, pointing away from the gripper.

    Args:
        sim (MjSim): Current simulation object.
        geoms_1 (str or list of str or MujocoModel or None): First geom group filter. If None, no filter
            is applied (all contacts are returned). If a MujocoModel, its ``contact_geoms`` are used.
        geoms_2 (str or list of str or MujocoModel or None): Second geom group filter. If None while
            @geoms_1 is set, any contact involving @geoms_1 (against any other geom) is returned.
        actor_geoms (str or list of str or MujocoModel or None): If provided, the returned ``force`` and
            ``normal`` are oriented so they represent the force the *actor* exerts on its contact partner
            (pointing away from the actor, into the partner). For a contact where an actor geom is
            ``geom2``, the raw force (which acts on ``geom2``) is negated; where it is ``geom1`` the raw
            force is kept. Pass the gripper here to get "force the gripper applies to the object".
        normalize_normal (bool): If True, the returned ``normal`` is unit length (it should already be,
            but floating point can drift slightly).

    Returns:
        list of dict: One entry per matching contact, each with keys:

            :`'geoms'`: ``(geom1_name, geom2_name)`` tuple (names may be None for unnamed geoms)
            :`'geom_ids'`: ``(geom1_id, geom2_id)`` tuple
            :`'pos'`: (3,) contact position in world frame
            :`'normal'`: (3,) unit contact normal in world frame, oriented along ``force``
            :`'force'`: (3,) full contact force vector in world frame (normal + friction); on ``geom2``
                by default, or on the actor's partner if @actor_geoms is given
            :`'normal_force'`: float, magnitude of the force along the normal (>= 0, compression)
            :`'force_mag'`: float, magnitude of the full force vector
            :`'wrench_contact'`: (6,) raw ``mj_contactForce`` wrench [fx, fy, fz, tx, ty, tz] in
                the contact frame (never sign-flipped)
    """
    m, d = sim.model._model, sim.data._data
    names_1 = _to_geom_name_list(geoms_1)
    names_2 = _to_geom_name_list(geoms_2)
    actor_set = set(_to_geom_name_list(actor_geoms) or [])

    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = sim.model.geom_id2name(c.geom1)
        g2 = sim.model.geom_id2name(c.geom2)

        # Apply the (symmetric) geom-group filter, matching check_contact's semantics.
        if names_1 is not None:
            c1_in_g1 = g1 in names_1
            c2_in_g1 = g2 in names_1
            c1_in_g2 = (g1 in names_2) if names_2 is not None else True
            c2_in_g2 = (g2 in names_2) if names_2 is not None else True
            if not ((c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1)):
                continue

        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(m, d, i, wrench)

        # Rows of contact.frame are the contact axes (normal, tangent1, tangent2) in world coords,
        # so frame.T maps contact-frame components back into the world frame. The raw force acts on
        # geom2 (points from geom1 toward geom2).
        frame = np.array(c.frame).reshape(3, 3)
        normal = frame[0].copy()
        if normalize_normal:
            n = np.linalg.norm(normal)
            if n > 0:
                normal = normal / n
        force = frame.T @ wrench[:3]

        # Re-orient so the force is what the actor exerts on its partner (away from the actor). Raw
        # force acts on geom2, so keep it when the actor is geom1, flip it when the actor is geom2.
        if actor_set and (g1 not in actor_set) and (g2 in actor_set):
            force = -force
            normal = -normal

        out.append(
            {
                "geoms": (g1, g2),
                "geom_ids": (int(c.geom1), int(c.geom2)),
                "pos": np.array(c.pos),
                "normal": normal,
                "force": force,
                "normal_force": float(wrench[0]),
                "force_mag": float(np.linalg.norm(force)),
                "wrench_contact": wrench,
            }
        )
    return out


def get_total_contact_force(sim, geoms_1=None, geoms_2=None, actor_geoms=None):
    """
    Convenience wrapper that sums the per-contact world-frame force vectors returned by
    :func:`get_contact_forces` into a single resultant force, along with its point of application
    (force-magnitude-weighted average contact position).

    Args:
        sim (MjSim): Current simulation object.
        geoms_1, geoms_2, actor_geoms: See :func:`get_contact_forces`.

    Returns:
        tuple:
            - net_force (np.array): (3,) summed force vector in world frame (zeros if no contacts).
            - application_point (np.array or None): (3,) magnitude-weighted contact position, or None
              if there are no contacts.
    """
    contacts = get_contact_forces(sim, geoms_1=geoms_1, geoms_2=geoms_2, actor_geoms=actor_geoms)
    if len(contacts) == 0:
        return np.zeros(3), None
    net_force = np.sum([c["force"] for c in contacts], axis=0)
    weights = np.array([c["force_mag"] for c in contacts])
    total = weights.sum()
    if total > 0:
        application_point = np.sum([w * c["pos"] for w, c in zip(weights, contacts)], axis=0) / total
    else:
        application_point = np.mean([c["pos"] for c in contacts], axis=0)
    return net_force, application_point


def aggregate_contact_forces(sim, contacts, actor_geoms=None, min_force=0.0, separate_by_actor=True):
    """
    Collapses a list of per-contact force dicts (from :func:`get_contact_forces`) into summed force
    groups. A single grasp/push produces several redundant contact points at the same interface, each
    carrying a noisy fraction of the total force; summing them yields a single, much steadier force
    vector and application point per group. This is spatial denoising, complementary to any temporal
    smoothing the caller applies across frames.

    Grouping granularity matters for grasps. If you sum *all* forces on a grasped object into one
    vector, the two fingers' forces (which push from opposite sides) nearly cancel, so the net is ~0 N
    and disappears below any threshold -- a grasp wrongly shows no force. To avoid this, by default
    (@separate_by_actor=True with @actor_geoms set) forces are grouped per **(contacted body, actor
    body)** pair, keeping each finger's force separate. A grasp then renders as two opposing "squeeze"
    arrows; a one-sided push stays a single arrow. Set @separate_by_actor=False to sum per contacted
    body only (use when you specifically want the *net* force on an object, e.g. for a push).

    The "partner" geom of a contact is the one *not* in @actor_geoms (e.g. the object the gripper
    touches); the "actor" geom is the one in @actor_geoms. If @actor_geoms is None, the body of
    ``geom1`` is the partner and grouping is per partner body regardless of @separate_by_actor.

    Args:
        sim (MjSim): Current simulation object.
        contacts (list of dict): Output of :func:`get_contact_forces` (forces should already be
            oriented via that function's ``actor_geoms`` if a consistent sign is desired).
        actor_geoms (str or list of str or MujocoModel or None): Geoms whose body is the actor; the
            partner body (the other geom) anchors each group.
        min_force (float): Drop groups whose summed force magnitude is below this (Newtons). Useful as
            a deadband against tiny jitter contacts.
        separate_by_actor (bool): If True (and @actor_geoms given), group per (partner body, actor
            body) pair so opposing-finger grasp forces do not cancel. If False, group per partner body.

    Returns:
        list of dict: One entry per group, each with keys:

            :`'key'`: hashable group key -- ``(partner_body_id, actor_body_id)`` when separating by
                actor, else ``partner_body_id``. Stable across frames; use it to track a group over time.
            :`'body_id'`: int body id of the contacted (partner) body
            :`'body_name'`: str name of that body (may be None)
            :`'actor_body_id'`: int body id of the actor body for this group (None if not separating)
            :`'actor_body_name'`: str name of the actor body (None if not separating)
            :`'force'`: (3,) summed world-frame force
            :`'pos'`: (3,) magnitude-weighted application point
            :`'force_mag'`: float magnitude of the summed force
            :`'num_contacts'`: int number of raw contacts that were merged
            :`'geoms'`: set of partner geom names included in this group
            :`'actor_geoms'`: set of actor geom names involved (e.g. which gripper geoms/fingerpads
                made the contact -- use to distinguish a fingerpad grasp from a body push)
    """
    actor_set = set(_to_geom_name_list(actor_geoms) or [])
    groups = {}
    for c in contacts:
        g1, g2 = c["geoms"]
        id1, id2 = c["geom_ids"]
        # Pick partner (non-actor) and actor geom ids/names. Default to geom1 as partner if no actor set.
        if actor_set and g1 in actor_set and g2 not in actor_set:
            partner_id, partner_name, actor_id, actor_name = id2, g2, id1, g1
        elif actor_set and g2 in actor_set and g1 not in actor_set:
            partner_id, partner_name, actor_id, actor_name = id1, g1, id2, g2
        else:
            partner_id, partner_name, actor_id, actor_name = id1, g1, None, None

        partner_body = int(sim.model.geom_bodyid[partner_id])
        actor_body = int(sim.model.geom_bodyid[actor_id]) if (actor_id is not None) else None
        if separate_by_actor and actor_body is not None:
            key = (partner_body, actor_body)
        else:
            key = partner_body
            actor_body = None

        g = groups.setdefault(
            key,
            {
                "partner_body": partner_body,
                "actor_body": actor_body,
                "force": np.zeros(3),
                "wpos": np.zeros(3),
                "wsum": 0.0,
                "num_contacts": 0,
                "geoms": set(),
                "actor_geoms": set(),
            },
        )
        g["force"] += c["force"]
        w = c["force_mag"]
        g["wpos"] += w * c["pos"]
        g["wsum"] += w
        g["num_contacts"] += 1
        g["geoms"].add(partner_name)
        if actor_name is not None:
            g["actor_geoms"].add(actor_name)

    out = []
    for key, g in groups.items():
        mag = float(np.linalg.norm(g["force"]))
        if mag < min_force:
            continue
        pos = g["wpos"] / g["wsum"] if g["wsum"] > 0 else g["wpos"]
        actor_body = g["actor_body"]
        out.append(
            {
                "key": key,
                "body_id": g["partner_body"],
                "body_name": sim.model.body_id2name(g["partner_body"]),
                "actor_body_id": actor_body,
                "actor_body_name": sim.model.body_id2name(actor_body) if actor_body is not None else None,
                "force": g["force"],
                "pos": pos,
                "force_mag": mag,
                "num_contacts": g["num_contacts"],
                "geoms": g["geoms"],
                "actor_geoms": g["actor_geoms"],
            }
        )
    return out


def get_sensor_measurement(sim, sensor_name):
    """
    Reads a named MuJoCo sensor's measurement from ``sim.data.sensordata``. Useful for the wrist
    force/torque sensors that ship with robosuite grippers (Level-1 net wrench at the wrist), whose
    names are exposed via ``gripper.important_sensors`` (e.g. ``"force_ee"`` / ``"torque_ee"`` keys map
    to the actual prefixed sensor names).

    Args:
        sim (MjSim): Current simulation object.
        sensor_name (str): Name of the sensor as registered in the MuJoCo model.

    Returns:
        np.array: The sensor's measurement (length depends on the sensor type, e.g. 3 for force/torque).
    """
    m, d = sim.model._model, sim.data._data
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid == -1:
        raise ValueError(
            "Sensor '{}' not found. Available sensors: {}".format(sensor_name, list(sim.model.sensor_names))
        )
    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
    return np.array(d.sensordata[adr : adr + dim])
