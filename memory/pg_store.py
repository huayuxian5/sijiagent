"""PostgreSQL 长期记忆存储。

自动建表，提供 episodic / preference / procedural 三类记忆的读写接口。
连接失败时静默降级为只读（不阻塞主流程）。使用连接池管理连接。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("agent.memory.pg_store")


def _decode_memory_value(value: Any, *, driver_id: str, key: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if not (
        text.startswith("{")
        or text.startswith("[")
        or (text.startswith('"') and text.endswith('"'))
        or text in {"true", "false", "null"}
    ):
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("pg memory value is not valid json; keeping raw string: driver_id=%s key=%s", driver_id, key)
        return value


class PgLongTermMemoryStore:
    """PostgreSQL-backed long-term memory store with connection pooling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "agent_memory",
        user: str = "postgres",
        password: str = "postgres",
        min_conn: int = 1,
        max_conn: int = 4,
    ) -> None:
        self._conn_params = {
            "host": host,
            "port": port,
            "dbname": database,
            "user": user,
            "password": password,
        }
        self._pool: Any = None
        self._available = False
        self._min_conn = min_conn
        self._max_conn = max_conn
        self._connect()

    def _connect(self) -> None:
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
            self._pool = ThreadedConnectionPool(
                self._min_conn,
                self._max_conn,
                **self._conn_params,
            )
            self._available = True
            self._init_tables()
            logger.info("PostgreSQL connected: %s:%s/%s", self._conn_params["host"], self._conn_params["port"], self._conn_params["dbname"])
        except Exception:
            logger.warning("PostgreSQL unavailable, long-term memory will be in-memory only", exc_info=True)
            self._available = False

    def _get_conn(self) -> Any:
        if self._pool is None:
            return None
        return self._pool.getconn()

    def _put_conn(self, conn: Any) -> None:
        if self._pool is not None and conn is not None:
            self._pool.putconn(conn)

    def _init_tables(self) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_episodic_memory (
                        id SERIAL PRIMARY KEY,
                        driver_id VARCHAR(32) NOT NULL,
                        event_type VARCHAR(32) NOT NULL DEFAULT 'take_order',
                        cargo_id VARCHAR(64),
                        score DOUBLE PRECISION,
                        net_income DOUBLE PRECISION,
                        target_lat DOUBLE PRECISION,
                        target_lng DOUBLE PRECISION,
                        simulation_minute INTEGER,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_episodic_driver
                    ON agent_episodic_memory (driver_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS agent_preference_memory (
                        driver_id VARCHAR(32) NOT NULL,
                        key VARCHAR(64) NOT NULL,
                        value JSONB,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (driver_id, key)
                    );

                    CREATE TABLE IF NOT EXISTS agent_hotspot_memory (
                        driver_id VARCHAR(32) NOT NULL,
                        lat_rounded DOUBLE PRECISION NOT NULL,
                        lng_rounded DOUBLE PRECISION NOT NULL,
                        weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (driver_id, lat_rounded, lng_rounded)
                    );

                    CREATE TABLE IF NOT EXISTS agent_pickup_hotspot_memory (
                        driver_id VARCHAR(32) NOT NULL,
                        lat_rounded DOUBLE PRECISION NOT NULL,
                        lng_rounded DOUBLE PRECISION NOT NULL,
                        weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (driver_id, lat_rounded, lng_rounded)
                    );

                    CREATE TABLE IF NOT EXISTS agent_reposition_failure (
                        driver_id VARCHAR(32) NOT NULL,
                        lat_rounded DOUBLE PRECISION NOT NULL,
                        lng_rounded DOUBLE PRECISION NOT NULL,
                        fail_count INTEGER NOT NULL DEFAULT 1,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (driver_id, lat_rounded, lng_rounded)
                    );

                    CREATE TABLE IF NOT EXISTS agent_cargo_density (
                        driver_id VARCHAR(32) NOT NULL,
                        lat_rounded DOUBLE PRECISION NOT NULL,
                        lng_rounded DOUBLE PRECISION NOT NULL,
                        cargo_count INTEGER NOT NULL DEFAULT 0,
                        avg_pickup_km DOUBLE PRECISION,
                        best_net_income DOUBLE PRECISION,
                        query_minute INTEGER NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (driver_id, lat_rounded, lng_rounded)
                    );
                """)
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("Failed to init pg tables", exc_info=True)
        finally:
            self._put_conn(conn)

    @property
    def available(self) -> bool:
        return self._available

    # ---- Episodic memory ----

    def save_episodic(self, driver_id: str, event: dict[str, Any]) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO agent_episodic_memory
                       (driver_id, event_type, cargo_id, score, net_income, target_lat, target_lng, simulation_minute)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        driver_id,
                        event.get("event_type", "take_order"),
                        event.get("cargo_id"),
                        event.get("score"),
                        event.get("net_income"),
                        event.get("target_lat"),
                        event.get("target_lng"),
                        event.get("simulation_minute"),
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg save_episodic failed", exc_info=True)
        finally:
            self._put_conn(conn)

    def load_episodic(self, driver_id: str, limit: int = 20) -> list[dict[str, Any]]:
        if not self._available or self._pool is None:
            return []
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT event_type, cargo_id, score, net_income, target_lat, target_lng, simulation_minute, created_at
                       FROM agent_episodic_memory
                       WHERE driver_id = %s
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (driver_id, limit),
                )
                rows = cur.fetchall()
                return [
                    {
                        "event_type": r[0],
                        "cargo_id": r[1],
                        "score": r[2],
                        "net_income": r[3],
                        "target_lat": r[4],
                        "target_lng": r[5],
                        "simulation_minute": r[6],
                        "created_at": r[7].isoformat() if r[7] else None,
                    }
                    for r in rows
                ]
        except Exception:
            logger.error("pg load_episodic failed", exc_info=True)
            return []
        finally:
            self._put_conn(conn)

    # ---- Preference memory ----

    def save_preference(self, driver_id: str, key: str, value: Any) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            json_val = json.dumps(value, ensure_ascii=False, default=str)
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO agent_preference_memory (driver_id, key, value, updated_at)
                       VALUES (%s, %s, %s, NOW())
                       ON CONFLICT (driver_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
                    (driver_id, key, json_val),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg save_preference failed", exc_info=True)
        finally:
            self._put_conn(conn)

    def load_preferences(self, driver_id: str) -> dict[str, Any]:
        if not self._available or self._pool is None:
            return {}
        conn = self._get_conn()
        if conn is None:
            return {}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT key, value FROM agent_preference_memory WHERE driver_id = %s",
                    (driver_id,),
                )
                rows = cur.fetchall()
                result: dict[str, Any] = {}
                for key, val in rows:
                    result[key] = _decode_memory_value(val, driver_id=driver_id, key=str(key))
                return result
        except Exception:
            logger.error("pg load_preferences failed", exc_info=True)
            return {}
        finally:
            self._put_conn(conn)

    # ---- Hotspot memory ----

    def save_hotspot(self, driver_id: str, lat: float, lng: float, weight: float) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO agent_hotspot_memory (driver_id, lat_rounded, lng_rounded, weight, updated_at)
                       VALUES (%s, %s, %s, %s, NOW())
                       ON CONFLICT (driver_id, lat_rounded, lng_rounded)
                       DO UPDATE SET weight = agent_hotspot_memory.weight + EXCLUDED.weight, updated_at = NOW()""",
                    (driver_id, lat, lng, weight),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg save_hotspot failed", exc_info=True)
        finally:
            self._put_conn(conn)

    def load_hotspots(self, driver_id: str) -> dict[tuple[float, float], float]:
        if not self._available or self._pool is None:
            return {}
        conn = self._get_conn()
        if conn is None:
            return {}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT lat_rounded, lng_rounded, weight FROM agent_hotspot_memory WHERE driver_id = %s",
                    (driver_id,),
                )
                rows = cur.fetchall()
                return {(float(r[0]), float(r[1])): float(r[2]) for r in rows}
        except Exception:
            logger.error("pg load_hotspots failed", exc_info=True)
            return {}
        finally:
            self._put_conn(conn)

    # ---- Pickup hotspot memory ----

    def save_pickup_hotspot(self, driver_id: str, lat: float, lng: float, weight: float) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO agent_pickup_hotspot_memory (driver_id, lat_rounded, lng_rounded, weight, updated_at)
                       VALUES (%s, %s, %s, %s, NOW())
                       ON CONFLICT (driver_id, lat_rounded, lng_rounded)
                       DO UPDATE SET weight = agent_pickup_hotspot_memory.weight + EXCLUDED.weight, updated_at = NOW()""",
                    (driver_id, lat, lng, weight),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg save_pickup_hotspot failed", exc_info=True)
        finally:
            self._put_conn(conn)

    def load_pickup_hotspots(self, driver_id: str) -> dict[tuple[float, float], float]:
        if not self._available or self._pool is None:
            return {}
        conn = self._get_conn()
        if conn is None:
            return {}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT lat_rounded, lng_rounded, weight FROM agent_pickup_hotspot_memory WHERE driver_id = %s",
                    (driver_id,),
                )
                rows = cur.fetchall()
                return {(float(r[0]), float(r[1])): float(r[2]) for r in rows}
        except Exception:
            logger.error("pg load_pickup_hotspots failed", exc_info=True)
            return {}
        finally:
            self._put_conn(conn)

    # ---- Reposition failure memory ----

    def save_reposition_failure(self, driver_id: str, lat: float, lng: float) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO agent_reposition_failure (driver_id, lat_rounded, lng_rounded, fail_count, updated_at)
                       VALUES (%s, %s, %s, 1, NOW())
                       ON CONFLICT (driver_id, lat_rounded, lng_rounded)
                       DO UPDATE SET fail_count = agent_reposition_failure.fail_count + 1, updated_at = NOW()""",
                    (driver_id, lat, lng),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg save_reposition_failure failed", exc_info=True)
        finally:
            self._put_conn(conn)

    def load_reposition_failures(self, driver_id: str) -> dict[tuple[float, float], int]:
        if not self._available or self._pool is None:
            return {}
        conn = self._get_conn()
        if conn is None:
            return {}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT lat_rounded, lng_rounded, fail_count FROM agent_reposition_failure WHERE driver_id = %s",
                    (driver_id,),
                )
                rows = cur.fetchall()
                return {(float(r[0]), float(r[1])): int(r[2]) for r in rows}
        except Exception:
            logger.error("pg load_reposition_failures failed", exc_info=True)
            return {}
        finally:
            self._put_conn(conn)

    # ---- Cargo density ----

    def save_cargo_density(
        self,
        driver_id: str,
        lat: float,
        lng: float,
        cargo_count: int,
        avg_pickup_km: float | None,
        best_net_income: float | None,
        query_minute: int,
    ) -> None:
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO agent_cargo_density
                       (driver_id, lat_rounded, lng_rounded, cargo_count, avg_pickup_km, best_net_income, query_minute, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (driver_id, lat_rounded, lng_rounded)
                       DO UPDATE SET
                           cargo_count = EXCLUDED.cargo_count,
                           avg_pickup_km = EXCLUDED.avg_pickup_km,
                           best_net_income = EXCLUDED.best_net_income,
                           query_minute = EXCLUDED.query_minute,
                           updated_at = NOW()""",
                    (driver_id, lat, lng, cargo_count, avg_pickup_km, best_net_income, query_minute),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg save_cargo_density failed", exc_info=True)
        finally:
            self._put_conn(conn)

    def load_top_density(self, driver_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """返回密度最高的 N 个区域。"""
        if not self._available or self._pool is None:
            return []
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT lat_rounded, lng_rounded, cargo_count, avg_pickup_km, best_net_income, query_minute
                       FROM agent_cargo_density
                       WHERE driver_id = %s
                       ORDER BY cargo_count DESC
                       LIMIT %s""",
                    (driver_id, limit),
                )
                rows = cur.fetchall()
                return [
                    {
                        "lat": float(r[0]),
                        "lng": float(r[1]),
                        "cargo_count": int(r[2]),
                        "avg_pickup_km": round(float(r[3]), 1) if r[3] is not None else None,
                        "best_net_income": round(float(r[4]), 1) if r[4] is not None else None,
                        "query_minute": int(r[5]),
                    }
                    for r in rows
                ]
        except Exception:
            logger.error("pg load_top_density failed", exc_info=True)
            return []
        finally:
            self._put_conn(conn)

    def load_density_near(self, driver_id: str, lat: float, lng: float, radius_km: float = 50.0) -> list[dict[str, Any]]:
        """返回指定位置附近的密度记录。"""
        if not self._available or self._pool is None:
            return []
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                # 用简单的经纬度范围过滤（0.1度 ≈ 11km，radius_km/11 约等于需要的网格数）
                delta = max(radius_km / 111.0, 0.5)
                cur.execute(
                    """SELECT lat_rounded, lng_rounded, cargo_count, avg_pickup_km, best_net_income, query_minute
                       FROM agent_cargo_density
                       WHERE driver_id = %s
                         AND lat_rounded BETWEEN %s AND %s
                         AND lng_rounded BETWEEN %s AND %s
                       ORDER BY cargo_count DESC""",
                    (driver_id, lat - delta, lat + delta, lng - delta, lng + delta),
                )
                rows = cur.fetchall()
                return [
                    {
                        "lat": float(r[0]),
                        "lng": float(r[1]),
                        "cargo_count": int(r[2]),
                        "avg_pickup_km": round(float(r[3]), 1) if r[3] is not None else None,
                        "best_net_income": round(float(r[4]), 1) if r[4] is not None else None,
                        "query_minute": int(r[5]),
                    }
                    for r in rows
                ]
        except Exception:
            logger.error("pg load_density_near failed", exc_info=True)
            return []
        finally:
            self._put_conn(conn)

    # ---- Clear ----

    def clear(self, driver_id: str) -> None:
        """清空指定司机的全部长期记忆。"""
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_episodic_memory WHERE driver_id = %s", (driver_id,))
                cur.execute("DELETE FROM agent_preference_memory WHERE driver_id = %s", (driver_id,))
                cur.execute("DELETE FROM agent_hotspot_memory WHERE driver_id = %s", (driver_id,))
                cur.execute("DELETE FROM agent_pickup_hotspot_memory WHERE driver_id = %s", (driver_id,))
                cur.execute("DELETE FROM agent_reposition_failure WHERE driver_id = %s", (driver_id,))
                cur.execute("DELETE FROM agent_cargo_density WHERE driver_id = %s", (driver_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("pg clear failed for driver_id=%s", driver_id, exc_info=True)
        finally:
            self._put_conn(conn)

    def clear_all(self) -> None:
        """清空全部长期记忆（仿真开始时调用）。"""
        if not self._available or self._pool is None:
            return
        conn = self._get_conn()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE agent_episodic_memory")
                cur.execute("TRUNCATE agent_preference_memory")
                cur.execute("TRUNCATE agent_hotspot_memory")
                cur.execute("TRUNCATE agent_pickup_hotspot_memory")
                cur.execute("TRUNCATE agent_reposition_failure")
                cur.execute("TRUNCATE agent_cargo_density")
            conn.commit()
            logger.info("pg memory cleared (all tables truncated)")
        except Exception:
            conn.rollback()
            logger.error("pg clear_all failed", exc_info=True)
        finally:
            self._put_conn(conn)

    # ---- Lifecycle ----

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.closeall()
            except Exception:
                pass
            self._pool = None
            self._available = False

    def _reconnect(self) -> None:
        self.close()
        try:
            self._connect()
        except Exception:
            pass
