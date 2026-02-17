import type { GraphitiClient, GraphitiMessage } from '../client.ts';

export interface GraphitiIngestToolInput {
  groupId: string;
  messages: GraphitiMessage[];
}

export const graphitiIngestTool = async (
  client: GraphitiClient,
  input: GraphitiIngestToolInput,
): Promise<void> => {
  await client.ingestMessages(input.groupId, input.messages);
};
