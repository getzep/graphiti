export interface GraphitiSlashCommandResult {
  ok: boolean;
  message: string;
}

export const graphitiSlashStatus = (): GraphitiSlashCommandResult => ({
  ok: true,
  message: 'Graphiti plugin scaffold loaded.',
});
