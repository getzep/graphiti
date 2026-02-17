import type { GraphitiClient, GraphitiSearchResults } from '../client.ts';

export interface GraphitiSearchToolInput {
  query: string;
  groupIds?: string[];
}

export const graphitiSearchTool = async (
  client: GraphitiClient,
  input: GraphitiSearchToolInput,
): Promise<GraphitiSearchResults> => {
  return client.search(input.query, input.groupIds);
};
