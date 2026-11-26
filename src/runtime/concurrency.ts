type Release = () => void;

export class Semaphore {
  private active = 0;
  private readonly waiters = new Set<() => void>();

  constructor(readonly capacity: number) {
    if (!Number.isSafeInteger(capacity) || capacity < 1) {
      throw new RangeError('Concurrency limits must be positive safe integers');
    }
  }

  get idle(): boolean {
    return this.active === 0 && this.waiters.size === 0;
  }

  async acquire(signal: AbortSignal): Promise<Release> {
    signal.throwIfAborted();
    if (this.active < this.capacity) return this.takeSlot();
    return new Promise((resolve, reject) => {
      const grant = () => {
        this.waiters.delete(grant);
        signal.removeEventListener('abort', abort);
        resolve(this.takeSlot());
      };
      const abort = () => {
        this.waiters.delete(grant);
        reject(signal.reason);
      };
      this.waiters.add(grant);
      signal.addEventListener('abort', abort, { once: true });
    });
  }

  private takeSlot(): Release {
    this.active++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.active--;
      this.waiters.values().next().value?.();
    };
  }
}

/** Stop waiting without abandoning observation of the underlying promise. */
export function waitWithSignal<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    promise.then((value) => {
      signal.removeEventListener('abort', abort);
      resolve(value);
    }, (error: unknown) => {
      signal.removeEventListener('abort', abort);
      reject(error);
    });
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  });
}
