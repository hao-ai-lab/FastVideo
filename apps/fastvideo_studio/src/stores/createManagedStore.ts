type Subscriber<T> = (state: T) => void;

export interface ManagedStore<T> {
  subscribe: (callback: Subscriber<T>) => () => void;
  get: () => T;
  /**
   * Stable state used for server prerender and the hydration render on the
   * client (React's `getServerSnapshot`). Defaults to the initial state; pass
   * `serverState` explicitly when the initial state is derived from browser
   * state (e.g. localStorage), otherwise the server HTML and the client's
   * hydration snapshot can disagree.
   */
  getServerSnapshot: () => T;
  set: (nextState: T) => T;
  update: (updater: (state: T) => T) => T;
  patch: (partial: Partial<T> | ((state: T) => Partial<T>)) => T;
}

export function createManagedStore<T extends object>(
  initialState: T,
  serverState: T = initialState,
): ManagedStore<T> {
  let currentState = initialState;
  const subscribers = new Set<Subscriber<T>>();

  function notify() {
    subscribers.forEach((cb) => cb(currentState));
  }

  function set(nextState: T): T {
    currentState = nextState;
    notify();
    return currentState;
  }

  function update(updater: (state: T) => T): T {
    return set(updater(currentState));
  }

  function patch(partial: Partial<T> | ((state: T) => Partial<T>)): T {
    return update((state) => ({
      ...state,
      ...(typeof partial === 'function' ? partial(state) : partial),
    }));
  }

  return {
    subscribe(callback: Subscriber<T>) {
      subscribers.add(callback);
      return () => {
        subscribers.delete(callback);
      };
    },
    get: () => currentState,
    getServerSnapshot: () => serverState,
    set,
    update,
    patch,
  };
}
