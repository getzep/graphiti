export interface GraphitiCliStatus {
  plugin: string;
  healthy: boolean;
  detail: string;
}

export const graphitiCliStatus = (): GraphitiCliStatus => ({
  plugin: 'graphiti-openclaw',
  healthy: true,
  detail: 'Plugin scaffold is installed. Use /graphiti status in runtime for live checks.',
});
