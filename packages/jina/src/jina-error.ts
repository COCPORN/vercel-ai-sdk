import { create } from '@ai-sdk/provider/dist/provider';

export const jinaFailedResponseHandler = create.failedResponseHandler({
  // Jina errors are not yet well-documented.
  // This is a placeholder for future, more specific error handling.
});
