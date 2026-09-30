/**
 * Env flags that guard destructive or hour-long work.
 *
 * `Boolean(process.env.X)` is true for "false" and "0", so an operator who
 * disables a gate the obvious way would still trip it — and these gates stand in
 * front of `docker compose down` and an hour of nvcc.
 */
export const flag = (name: string): boolean => {
  const v = process.env[name]?.trim().toLowerCase();
  return v !== undefined && v !== '' && v !== '0' && v !== 'false' && v !== 'no';
};
