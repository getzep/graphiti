import fs from 'node:fs';
import path from 'node:path';

export const isPathWithinRoot = (root: string, target: string): boolean => {
  const relative = path.relative(root, target);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
};

export const toCanonicalPath = (candidate: string, label: string): string => {
  try {
    return fs.realpathSync(candidate);
  } catch (error) {
    throw new Error(`Unable to resolve ${label}: ${(error as Error).message}`);
  }
};
